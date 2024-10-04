from huggingface_hub import HfApi
print("nhập token")
a = input()
api = HfApi(token=a)
print("load")

api.upload_folder(
    folder_path="",
    repo_id="namkuner/Train_StyleTTS2_Vietnamese",
    repo_type="model",
)