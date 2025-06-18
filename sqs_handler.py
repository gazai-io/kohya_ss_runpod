from datetime import datetime
import os
import shutil
from typing import List, Optional
import boto3
from dotenv import load_dotenv
import toml
import time
import json
import random
from sqlalchemy.orm import Session

from kohya_gui import dreambooth_folder_creation_gui
from kohya_gui.common_gui import (
    get_executable_path,
    scriptdir,
    validate_folder_path,
    validate_model_path,
    setup_environment,
)
from kohya_gui.class_accelerate_launch import AccelerateLaunch
from kohya_gui.class_command_executor import CommandExecutor
from kohya_gui.custom_logging import setup_logging

from app.models import Image, LoraModelStatus, LoraModel
from app.database import SessionLocal

# Set up log
log = setup_logging()

load_dotenv()

PRETRAINED_MODEL_DIR = "/workspace/storage/stable_diffusion/models/ckpt"
LORA_DIR = "/workspace/storage/stable_diffusion/models/lora"

CLIP_L_PATH = "/workspace/storage/stable_diffusion/models/clip/clip_l.safetensors"
CLIP_G_PATH = "/workspace/storage/stable_diffusion/models/clip/clip_g.safetensors"
T5XXL_PATH = "/workspace/storage/stable_diffusion/models/clip/t5xxl_fp16.safetensors"
AE_PATH = "/workspace/storage/stable_diffusion/models/vae/ae.safetensors"
PROJECT_DIR = os.environ.get("PROJECT_DIR")

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.environ.get("AWS_REGION")

SQS_QUEUE_URL = os.environ.get("SQS_QUEUE_URL")
sqs = boto3.client(
    "sqs",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

BUCKET_NAME = "gazai"
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)


# Setup command executor
executor = None

def _folder_preparation(
    user_id: str,
    model_id: str,
    instance_prompt: str,
    class_prompt: str,
    training_images: List[Image],
    training_dir_output: str
):
    training_images_dir_input = rf"{PROJECT_DIR}/{user_id}/{model_id}/raw/img"
    os.makedirs(training_images_dir_input, exist_ok=True)

    for index, training_image in enumerate(training_images):
        object_key = training_image.image.objectKey
        if not object_key.startswith("assets"):
            object_key = rf"assets/{user_id}/{object_key}"

        file_extension = os.path.splitext(object_key)[1]
        local_img_file_path = os.path.join(
            training_images_dir_input, f"{index}{file_extension}"
        )

        s3.download_file(BUCKET_NAME, object_key, local_img_file_path)
        log.info(f"Downloaded {object_key} to {local_img_file_path}...")

        if training_image.caption is not None and training_image.caption != "":
            local_caption_file_path = os.path.join(
                training_images_dir_input, f"{index}.txt"
            )
            with open(local_caption_file_path, "a") as file:
                file.write(f"{training_image.caption}")
            log.info(
                f"Wrote {training_image.caption} to {local_caption_file_path}..."
            )

    dreambooth_folder_creation_gui.dreambooth_folder_preparation(
        util_training_images_dir_input=training_images_dir_input,
        util_training_images_repeat_input=40,
        util_instance_prompt_input=instance_prompt,
        util_regularization_images_dir_input="",
        util_regularization_images_repeat_input=1,
        util_class_prompt_input=class_prompt,
        util_training_dir_output=training_dir_output,
    )


def prepare_training_config_and_command(
    user_id: str,
    model_id: str,
    model_output_name: str,
    instance_prompt: str,
    class_prompt: str,
    training_images: List[Image],
    network_type='sdxl',
    pretrained_model='sd_xl_base_1.0.safetensors'
    ):

    #
    # Validate paths
    #

    pretrained_model_name_or_path = rf"{PRETRAINED_MODEL_DIR}/{pretrained_model}"
    if not validate_model_path(pretrained_model_name_or_path):
        return

    training_project_dir = rf"{PROJECT_DIR}/{user_id}/{model_id}"
    os.makedirs(training_project_dir, exist_ok=True)

    training_dir_output = os.path.join(training_project_dir, "output")
    _folder_preparation(
        user_id=user_id,
        model_id=model_id,
        instance_prompt=instance_prompt,
        class_prompt=class_prompt,
        training_images=training_images,
        training_dir_output=training_dir_output,
    )

    log_dir = os.path.join(training_dir_output, "log")
    training_images_dir_input = os.path.join(training_dir_output, "img")
    model_output_dir = os.path.join(training_dir_output, "model")
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    if not validate_folder_path(
        log_dir, can_be_written_to=True, create_if_not_exists=True
    ):
        return

    if not validate_folder_path(training_images_dir_input):
        return

    if not validate_folder_path(
        model_output_dir, can_be_written_to=True, create_if_not_exists=True
    ):
        return

    #
    # End of path validation
    #

    if training_images_dir_input == "":
        log.error("Training images directory is empty")
        return

    # Get a list of all subfolders in train_data_dir
    subfolders = [
        f
        for f in os.listdir(training_images_dir_input)
        if os.path.isdir(os.path.join(training_images_dir_input, f))
    ]

    total_steps = 0

    # Loop through each subfolder and extract the number of repeats
    for folder in subfolders:
        try:
            # Extract the number of repeats from the folder name
            repeats = int(folder.split("_")[0])
            log.info(f"Folder {folder}: {repeats} repeats found")

            # Count the number of images in the folder
            num_images = len(
                [
                    f
                    for f, lower_f in (
                        (file, file.lower())
                        for file in os.listdir(os.path.join(training_images_dir_input, folder))
                    )
                    if lower_f.endswith((".jpg", ".jpeg", ".png", ".webp"))
                ]
            )

            log.info(f"Folder {folder}: {num_images} images found")

            # Calculate the total number of steps for this folder
            steps = repeats * num_images

            # log.info the result
            log.info(f"Folder {folder}: {num_images} * {repeats} = {steps} steps")

            total_steps += steps

        except ValueError:
            # Handle the case where the folder name does not contain an underscore
            log.info(
                f"Error: '{folder}' does not contain an underscore, skipping..."
            )


    accelerate_path = get_executable_path("accelerate")
    if accelerate_path == "":
        log.error("accelerate not found")
        return

    run_cmd = [rf"{accelerate_path}", "launch"]

    run_cmd = AccelerateLaunch.run_cmd(
        run_cmd=run_cmd,
        dynamo_backend="no",
        dynamo_mode="default",
        dynamo_use_fullgraph=False,
        dynamo_use_dynamic=False,
        num_processes=1,
        num_machines=1,
        multi_gpu=False,
        gpu_ids="",
        main_process_port=0,
        num_cpu_threads_per_process=2,
        mixed_precision="bf16" if network_type == 'flux1' or network_type == 'sd3' else "fp16",
        extra_accelerate_launch_args="",
    )

    network_module = "networks.lora"
    if network_type == 'sdxl':
        run_cmd.append(rf"{scriptdir}/sd-scripts/sdxl_train_network.py")
    elif network_type == 'flux1':
        run_cmd.append(rf"{scriptdir}/sd-scripts/flux_train_network.py")
        network_module = "networks.lora_flux"
    elif network_type == 'sd3':
        run_cmd.append(rf"{scriptdir}/sd-scripts/sd3_train_network.py")
        network_module = "networks.lora_sd3"
    else:
        run_cmd.append(rf"{scriptdir}/sd-scripts/train_network.py")

    seed = random.randint(1, 2**32 - 1)

    config_toml_data = {
        # common settings
        "enable_bucket": True,
        "caption_extension": ".txt",
        "bucket_no_upscale": True,
        "bucket_reso_steps": 64,
        "cache_latents": True,
        "cache_latents_to_disk": True,
        "min_bucket_reso": 256,
        "max_bucket_reso": 2048,
        "resolution": "1024,1024",
        "max_train_epochs": 6,
        "xformers": True,
        "clip_skip": 2, # 2 for 2D character, 1 for real life character
        "network_dim": 16,
        "network_alpha": 16,
        "shuffle_caption": True,
        "prior_loss_weight": 1,
        "train_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": True,
        "learning_rate": 0.0001,
        "unet_lr": 0.0001,
        "text_encoder_lr": 0.00005,
        "save_model_as": "safetensors",
        "optimizer_type": "AdamW8bit",
        "optimizer_args": ["weight_decay=0.01", "betas=0.9,0.999", "eps=0.000001"],
        "lr_scheduler": "cosine_with_restarts",
        "max_token_length": int(150) if not network_type == 'flux1' else None,
        "network_module": network_module,
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "train_data_dir": training_images_dir_input,
        "output_dir": model_output_dir,
        "output_name": model_output_name,
        "save_every_n_epochs": 2,
        "seed": int(seed) if int(seed) != 0 else None,


        "save_precision": "fp16",
        "network_train_unet_only": True if network_type == 'flux1' or network_type == 'sd3' else False,
        "vae": AE_PATH if network_type == 'flux1' else None,
        "clip_l": CLIP_L_PATH if network_type == 'flux1' or network_type == 'sd3' else None,
        "clip_g": CLIP_G_PATH if network_type == 'sd3' else None,
        "t5xxl": T5XXL_PATH if network_type == 'flux1' or network_type == 'sd3' else None,
        "t5xxl_max_token_length": int(154) if network_type == 'flux1' or network_type == 'sd3' else None,
        "discrete_flow_shift": float(3.0) if network_type == 'flux1' else None,
        "model_prediction_type": "raw" if network_type == 'flux1' else None,
        "timestep_sampling": "shift" if network_type == 'flux1' else None,
        "apply_t5_attn_mask": True if network_type == 'flux1' or network_type == 'sd3' else None,
        "guidance_scale": float(1.0) if network_type == 'flux1' or network_type == 'sd3' else None,
        "cache_text_encoder_outputs": True if network_type == 'flux1' or network_type == 'sd3' else None,
        "cache_text_encoder_outputs_to_disk": True if network_type == 'flux1' or network_type == 'sd3' else None,
        "fp8_base": True if network_type == 'flux1' or network_type == 'sd3' else None,
        "huber_c": float(0.1) if network_type == 'flux1' or network_type == 'sd3' else None,
        "huber_schedule": "snr" if network_type == 'flux1' or network_type == 'sd3' else None,
        "loss_type": "l2" if network_type == 'flux1' or network_type == 'sd3' else None,
        "lr_scheduler_num_cycles": int(1) if network_type == 'flux1' or network_type == 'sd3' else None,
        "lr_scheduler_power": int(1) if network_type == 'flux1' or network_type == 'sd3' else None,
        "max_grad_norm": float(0.01) if network_type == 'flux1' or network_type == 'sd3' else None,
        "max_timestep": int(1000) if network_type == 'flux1' or network_type == 'sd3' else None,
    }

    # Given dictionary `config_toml_data`
    # Remove all values = ""
    config_toml_data = {
        key: value
        for key, value in config_toml_data.items()
        if value not in ["", False, None]
    }

    config_toml_data["max_data_loader_n_workers"] = int(0)



    # Sort the dictionary by keys
    config_toml_data = dict(sorted(config_toml_data.items()))



    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d-%H%M%S")
    tmpfilename = rf"{model_output_dir}/config_lora-{formatted_datetime}.toml"

    # Save the updated TOML data back to the file
    with open(tmpfilename, "w", encoding="utf-8") as toml_file:
        toml.dump(config_toml_data, toml_file)

        if not os.path.exists(toml_file.name):
            log.error(f"Failed to write TOML file: {toml_file.name}")

    run_cmd.append("--config_file")
    run_cmd.append(rf"{tmpfilename}")

    log.info(run_cmd)
    env = setup_environment()


    return config_toml_data, run_cmd, env


def train_lora_model(model_data: dict):
    log.info("train_lora_model task started...")
    database: Optional[Session] = None
    db_lora_model: Optional[LoraModel] = None
    model_root_dir: Optional[str] = None

    try:
        database = SessionLocal()
        db_lora_model = (
            database.query(LoraModel).filter_by(id=model_data["id"]).first()
        )

        if not db_lora_model:
            log.error(f"LoraModel with id {model_data['id']} not found.")
            return

        model_root_dir = rf"{PROJECT_DIR}/{db_lora_model.userId}/{db_lora_model.id}"
        os.makedirs(model_root_dir, exist_ok=True)


        # Perform the actual model training
        training_dir_output = os.path.join(model_root_dir, "output")
        os.makedirs(training_dir_output, exist_ok=True)
        _folder_preparation(
            user_id=db_lora_model.userId,
            model_id=db_lora_model.id,
            instance_prompt=db_lora_model.instancePrompt,
            class_prompt=db_lora_model.classPrompt,
            training_images=db_lora_model.trainingImages,
            training_dir_output=training_dir_output,
        )

        if db_lora_model.baseModel == "FLUX.1 D":
            network_type = "flux1"
        elif db_lora_model.baseModel == "SD 3.5":
            network_type = "sd3"
        else:
            network_type = "sdxl"
        config_toml_data, run_cmd, env = prepare_training_config_and_command(
            user_id=db_lora_model.userId,
            model_id=db_lora_model.id,
            model_output_name=db_lora_model.fileName,
            instance_prompt=db_lora_model.instancePrompt,
            class_prompt=db_lora_model.classPrompt,
            training_images=db_lora_model.trainingImages,
            network_type=network_type,
            pretrained_model=db_lora_model.trainingBaseModel,
        )

        # Update the status to "training"
        db_lora_model.status = LoraModelStatus.TRAINING
        db_lora_model.error = None # Clear previous errors
        db_lora_model.trainingParameters = config_toml_data
        db_lora_model.trainingStartedAt = datetime.now()
        db_lora_model.trainingEndedAt = None
        database.commit()

        global executor
        if executor is None:
            executor = CommandExecutor(headless=True)

        if executor.is_running():
            log.error(
                "Training is already running. Can't start another training session."
            )
            db_lora_model.status = LoraModelStatus.ERROR
            db_lora_model.error = "Another training session is already running."
            db_lora_model.trainingEndedAt = datetime.now()
            database.commit()
            return

        # Store model ID before closing session and detaching db_lora_model
        # db_lora_model here is the instance from the session we are about to close.
        # model_data['id'] is from the input message.
        model_id_to_refetch = None
        if db_lora_model:
            model_id_to_refetch = db_lora_model.id
        elif "id" in model_data:
            model_id_to_refetch = model_data["id"]

        # Close database connection before long-running command
        log.info("Closing database connection before executing training command.")
        if database:
            database.close()
            database = None # Mark session as closed

        # Run the command
        executor.execute_command(run_cmd=run_cmd, env=env)
        executor.wait_for_training_to_end() # This might raise an exception if training fails

        # Re-open database connection and re-fetch the model object
        log.info("Training command finished. Re-opening database connection.")
        database = SessionLocal() # Assign to the main 'database' variable

        if not model_id_to_refetch:
            log.critical("Cannot determine model ID to re-fetch LoraModel. Aborting further database operations.")
            # This will likely lead to an error handled by the main except block or a direct failure.
            raise ValueError("Model ID not available for re-fetching LoraModel after command execution.")

        # Re-assign db_lora_model to the instance from the new session
        db_lora_model = (
            database.query(LoraModel).filter_by(id=model_id_to_refetch).first()
        )

        if not db_lora_model:
            log.error(
                f"LoraModel with id {model_id_to_refetch} not found after re-opening database. Cannot update status."
            )
            # This is a critical error. Subsequent access to db_lora_model attributes will raise AttributeError.
            # This will be caught by the (modified) general except block, which will attempt to set ERROR status.
            raise LookupError(f"Failed to re-fetch LoraModel with id {model_id_to_refetch} after command execution.")

        log.info("Training completed ...") # Log success after command and successful re-fetch

        # Update the status to "ready"
        model_output_dir = os.path.join(training_dir_output, "model")
        save_model_as = "safetensors"

        safetensor_files = [
            f
            for f in os.listdir(model_output_dir)
            if f.endswith(f".{save_model_as}")
        ]

        primary_model_to_copy = None

        if not safetensor_files:
            log.error("No safetensors model file found after training.")
            db_lora_model.status = LoraModelStatus.ERROR
            db_lora_model.error = (
                "Training finished but no model file was created."
            )
            db_lora_model.trainingEndedAt = datetime.now()
            database.commit()
            return

        uploaded_object_keys = []
        for filename in safetensor_files:
            model_file_path = os.path.join(model_output_dir, filename)
            object_key = (
                f"loras/{db_lora_model.userId}/{db_lora_model.id}/{filename}"
            )
            s3.upload_file(model_file_path, BUCKET_NAME, object_key)
            uploaded_object_keys.append(object_key)
            log.info(f"Uploaded {filename} to S3.")

        primary_model_filename = f"{db_lora_model.fileName}.{save_model_as}"

        if primary_model_filename not in safetensor_files:
            log.error(
                f"Primary model file {primary_model_filename} not found after training."
            )
            db_lora_model.status = LoraModelStatus.ERROR
            db_lora_model.error = (
                f"Training finished but final model file {primary_model_filename} was not created."
            )
            db_lora_model.trainingEndedAt = datetime.now()
            database.commit()
            return

        primary_object_key = next(
            (
                key
                for key in uploaded_object_keys
                if key.endswith(f"/{primary_model_filename}")
            ),
            None,
        )

        primary_model_to_copy = os.path.join(
            model_output_dir, primary_model_filename
        )

        db_lora_model.objectKey = primary_object_key
        db_lora_model.status = LoraModelStatus.READY
        db_lora_model.trainingEndedAt = datetime.now()
        database.commit()

        if primary_model_to_copy and os.path.exists(primary_model_to_copy):
            destination_path = os.path.join(
                LORA_DIR, f"{db_lora_model.fileName}.{save_model_as}"
            )
            shutil.copy(primary_model_to_copy, destination_path)
            log.info(f"Copied {primary_model_to_copy} to {destination_path}")
        else:
            log.error(
                f"Could not find model file at {primary_model_to_copy} to copy."
            )
            db_lora_model.status = LoraModelStatus.ERROR
            db_lora_model.error = (
                f"Could not find model file at {primary_model_to_copy} to copy."
            )
            database.commit()
            return

    except Exception as e:
        log.error(f"Error during training: {e}", exc_info=True)
        # Attempt to update the model status to ERROR using a new, dedicated database session.
        # This ensures that even if the main 'database' session (from the try block)
        # was closed or is in an unusable state, we make a best effort to record the error.
        error_update_session: Optional[Session] = None
        try:
            # Determine the model ID for error updating.
            # model_data['id'] is the most reliable source from the input message.
            # db_lora_model (if it exists) might be a detached instance or None.
            lora_model_id_for_error = model_data.get("id")

            # Fallback if model_data.get("id") is None for some reason, but we had an instance.
            # This 'db_lora_model' is the one from the outer scope.
            if not lora_model_id_for_error and db_lora_model:
                try:
                    # Accessing .id on a detached instance is usually safe for simple attributes.
                    lora_model_id_for_error = db_lora_model.id
                except Exception as id_access_exc:
                    log.warning(f"Could not access .id from db_lora_model instance: {id_access_exc}")


            if lora_model_id_for_error:
                log.info(f"Attempting to update LoraModel {lora_model_id_for_error} to ERROR status.")
                error_update_session = SessionLocal()
                model_to_update_error_status = (
                    error_update_session.query(LoraModel)
                    .filter_by(id=lora_model_id_for_error)
                    .first()
                )
                if model_to_update_error_status:
                    model_to_update_error_status.status = LoraModelStatus.ERROR
                    model_to_update_error_status.error = str(e) # Record the original error 'e'
                    model_to_update_error_status.trainingEndedAt = datetime.now()
                    error_update_session.commit()
                    log.info(f"Successfully updated LoraModel {lora_model_id_for_error} to ERROR status.")
                else:
                    log.error(
                        f"Could not find LoraModel with id {lora_model_id_for_error} in database to update its error status."
                    )
            else:
                log.error(
                    "Could not determine LoraModel ID. Unable to update status to ERROR in database."
                )
        except Exception as inner_exception_during_error_handling:
            log.error(f"Further error occurred while trying to update LoraModel status to ERROR: {inner_exception_during_error_handling}", exc_info=True)
            if error_update_session:
                error_update_session.rollback() # Rollback any partial changes in the error session
        finally:
            if error_update_session:
                error_update_session.close() # Ensure the error session is closed
    finally:
        if database:
            database.close()
        if model_root_dir and os.path.exists(model_root_dir):
            shutil.rmtree(model_root_dir)
            log.info(f"Cleaned up training directory: {model_root_dir}")


def process_message(message):
    # Parse the message payload
    payload = json.loads(message["Body"])

    # Execute the LoRA training script
    train_lora_model(payload)


def main():
    log.info("Listening for messages...")
    while True:
        # Receive message from SQS queue
        response = sqs.receive_message(
            QueueUrl=SQS_QUEUE_URL, MaxNumberOfMessages=1, WaitTimeSeconds=20
        )

        # Check if a message was received
        if "Messages" in response:
            for message in response["Messages"]:
                try:
                    process_message(message)

                    # Delete the message from the queue
                    sqs.delete_message(
                        QueueUrl=SQS_QUEUE_URL, ReceiptHandle=message["ReceiptHandle"]
                    )
                except Exception as e:
                    log.error(f"Error processing message: {e}")

        # Optional: Add a small delay to avoid excessive polling
        time.sleep(1)


if __name__ == "__main__":
    main()
