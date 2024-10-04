# Fine Turning StyleTTS on VietNamese Dataset
## Giải thích hàm main()
Lấy config tu file config_path = Configs/config_ft.yml

    config = yaml.safe_load(open(config_path))

Lấy nơi lưu trữ của log trong config  

    log_dir = config['log_dir']

Nếu như log_dir chưa tồn tại thì tạo nó 

    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)

Copy file config sang thư mục log : Điều này dùng để đánh dấu tất cả tài nguyên được tạo ra 
    
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))

Khởi tạo writer và log
    
    writer = SummaryWriter(log_dir + "/tensorboard")
    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)

Lấy giá trị **batch size** từ config, nếu không tồn  tại giá trị batch size trong config thì mặc định là **10**

    batch_size = config.get('batch_size', 10)

Epoch : Số lần mà dữ liệu sẽ được lặp qua
    
    epochs = config.get('epochs', 200)

Tôi sẽ để code trước là giải thích sau :

    save_freq = config.get('save_freq', 2)

Tần suất lưu trạng thái của quá trình train (weight, learning rate schedule,...) mặc định là 2

    log_interval = config.get('log_interval', 10)

Tần suât lưu log , mặc định là 10 batch_size (step)

    data_params = config.get('data_params', None)

Thông tin của các tham số của dữ liệu

    sr = config['preprocess_params'].get('sr', 24000)

Lấy trong tham số của tiền xử lí dữ liệu, **sr** là **sampling rate**  
Sampling rate là tốc độ lấy mẫu âm thanh trong 1s  
Thông thường khi âm thanh được chuyển đổi thành dạng số thì nó là những tín hiệu có giá trị số. 
sample  = 24000 = 24 KHz có nghĩa là tôi sẽ lấy 1 giá trị số biểu diễn cường độ âm thanh tại trong thời gian 1/24000 s

    train_path = data_params['train_data']
    val_path = data_params['val_data']

Thư mục hướng tới dữ liệu huấn luyện và dữ liệu kiểm thử

    root_path = data_params['root_path']

Thư mục gốc của dữ liệu huấn luyện và kiểm thử

    min_length = data_params['min_length']

Giá trị nhỏ nhất của 1 đoạn âm thanh, nếu dữ liệu có đoạn âm thanh ngắn hơn **min_length** thì loại bỏ

    OOD_data = data_params['OOD_data']

Không rõ (Tra chatgpt). Nó có thể là dữ liệu test

    max_len = config.get('max_len', 200)

Độ dài lớn nhất của 1 đoạn âm thanh trong dữ liệu huấn luyện

    loss_params = Munch(config['loss_params'])
    
Các tham số của hàm loss (weight decay, epsilon, moteum,...) . Nó được chạy qua đối tượng Munch.


---------------------
Đối tượng Munch trong python giúp cho các tham số dạng json biến nó thành 1 biến. VD

    {
      "loss_params": {
        "loss_function": "cross_entropy",
        "weight_decay": 0.01
      }
    }
Chúng ta có thể gọi nó như sau

    from munch import Munch
    import json
    
    # Đọc tệp cấu hình (giả sử là tệp JSON)
    with open('config.json', 'r') as file:
        config = json.load(file)
    
    # Chuyển đổi phần 'loss_params' thành Munch
    loss_params = Munch(config['loss_params'])
    
    # Sử dụng các tham số mất mát
    print(loss_params.loss_function)  # Output: cross_entropy
    print(loss_params.weight_decay)   # Output: 0.01

---------------------------------

    diff_epoch = loss_params.diff_epoch
    joint_epoch = loss_params.joint_epoch

diff_epoch sẽ thực hiện 1 số thay đổi về cách huấn luyện (lr, batchsize,...) trong 1 số epoch được chỉ định  
join_epoch sẽ chỉ định 1 số epoch mà lúc đó các mô hình được kết hợp và phối hợp huấn luyện  
Nói chung 2 thằng này chưa rõ.

    optimizer_params = Munch(config['optimizer_params'])

Lấy các giá trị tham số của hàm tối ưu hóa

    train_list, val_list = get_data_path_list(train_path, val_path)

Là kết quả của hàm get_data_path_list

------------------------
Hàm get_data_path_list

    def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_list = f.readlines()
    with open(val_path, 'r', encoding='utf-8', errors='ignore') as f:
        val_list = f.readlines()

    return train_list, val_list

Mục đích là dùng để đọc tất cả các record (data)

-----------------------------------

    device = 'cuda'

Dùng GPU để train 


    train_dataloader = build_dataloader(train_list,
                                        root_path,
                                        OOD_data=OOD_data,
                                        min_length=min_length,
                                        batch_size=batch_size,
                                        num_workers=2,
                                        dataset_config={},
                                        device=device)

    val_dataloader = build_dataloader(val_list,
                                      root_path,
                                      OOD_data=OOD_data,
                                      min_length=min_length,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=0,
                                      device=device,
                                      dataset_config={})

Load dataset: chi tiết hàm [**Build Dataloader**](README/Build_dataloader.md)

    # load pretrained ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

Load ASR model : Automatic Speech Recognition là model dùng để phát hiện , căn chỉnh các âm thanh.  

    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)

F0 model dùng để tính toán độ cao pitch cho âm thanh, từ đó làm cho giọng nói tự nhiên hơn.  

    # load PL-BERT model
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)

PL BERT là model cực kì quan trọng dùng để dự đoán các âm vực đã bị ẩn đi trong hoặc model không phát hiện được trong giọng nói.

    model_params = recursive_munch(config['model_params'])

Load các trọng số của model trong config

    multispeaker = model_params.multispeaker

Đào tạo theo hướng nhiều người nói hoặc một người nói, với những cách khác nhau model sẽ khác nhau  

    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].to(device) for key in model]

Hàm build model này dùng để xây dựng model để train , khá là quan trọng nên tôi sẽ giải thích ở file riêng


