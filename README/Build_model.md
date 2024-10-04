# Chi tiết về hàm build model   
    def build_model(args, text_aligner, pitch_extractor, bert):
        assert args.decoder.type in ['istftnet', 'hifigan'], 'Decoder type unknown'
        
        if args.decoder.type == "istftnet":
            from Modules.istftnet import Decoder
            decoder = Decoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                    resblock_kernel_sizes = args.decoder.resblock_kernel_sizes,
                    upsample_rates = args.decoder.upsample_rates,
                    upsample_initial_channel=args.decoder.upsample_initial_channel,
                    resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                    upsample_kernel_sizes=args.decoder.upsample_kernel_sizes, 
                    gen_istft_n_fft=args.decoder.gen_istft_n_fft, gen_istft_hop_size=args.decoder.gen_istft_hop_size) 
        else:
            from Modules.hifigan import Decoder
            decoder = Decoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                    resblock_kernel_sizes = args.decoder.resblock_kernel_sizes,
                    upsample_rates = args.decoder.upsample_rates,
                    upsample_initial_channel=args.decoder.upsample_initial_channel,
                    resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                    upsample_kernel_sizes=args.decoder.upsample_kernel_sizes) 
            
        text_encoder = TextEncoder(channels=args.hidden_dim, kernel_size=5, depth=args.n_layer, n_symbols=args.n_token)
        
        predictor = ProsodyPredictor(style_dim=args.style_dim, d_hid=args.hidden_dim, nlayers=args.n_layer, max_dur=args.max_dur, dropout=args.dropout)
        
        style_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim) # acoustic style encoder
        predictor_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim) # prosodic style encoder
            
        # define diffusion model
        if args.multispeaker:
            transformer = StyleTransformer1d(channels=args.style_dim*2, 
                                        context_embedding_features=bert.config.hidden_size,
                                        context_features=args.style_dim*2, 
                                        **args.diffusion.transformer)
        else:
            transformer = Transformer1d(channels=args.style_dim*2, 
                                        context_embedding_features=bert.config.hidden_size,
                                        **args.diffusion.transformer)
        
        diffusion = AudioDiffusionConditional(
            in_channels=1,
            embedding_max_length=bert.config.max_position_embeddings,
            embedding_features=bert.config.hidden_size,
            embedding_mask_proba=args.diffusion.embedding_mask_proba, # Conditional dropout of batch elements,
            channels=args.style_dim*2,
            context_features=args.style_dim*2,
        )
        
        diffusion.diffusion = KDiffusion(
            net=diffusion.unet,
            sigma_distribution=LogNormalDistribution(mean = args.diffusion.dist.mean, std = args.diffusion.dist.std),
            sigma_data=args.diffusion.dist.sigma_data, # a placeholder, will be changed dynamically when start training diffusion model
            dynamic_threshold=0.0 
        )
        diffusion.diffusion.net = transformer
        diffusion.unet = transformer
    
        
        nets = Munch(
                bert=bert,
                bert_encoder=nn.Linear(bert.config.hidden_size, args.hidden_dim),
    
                predictor=predictor,
                decoder=decoder,
                text_encoder=text_encoder,
    
                predictor_encoder=predictor_encoder,
                style_encoder=style_encoder,
                diffusion=diffusion,
    
                text_aligner = text_aligner,
                pitch_extractor=pitch_extractor,
    
                mpd = MultiPeriodDiscriminator(),
                msd = MultiResSpecDiscriminator(),
            
                # slm discriminator head
                wd = WavLMDiscriminator(args.slm.hidden, args.slm.nlayers, args.slm.initial_channel),
           )
        
        return nets

Đầu tiên ta cùng tham khảo các tham số được truyền vào :  

    build_model(args, text_aligner, pitch_extractor, bert)

**args** :tham số trong config.  
**text_aligner** : là model ASR, dùng để nhận dạng giọng nói tự động.
**pitch_extractor** : là model trích xuất độ cao giọng nói.  
**bert** : là model dự đoán âm vị.  

    assert args.decoder.type in ['istftnet', 'hifigan'], 'Decoder type unknown'

Kiểm tra xem định dạng của mã hóa args.decoder.type  có nằm trong ['istftnet', 'hifigan'] hay không, nếu không thì báo lỗi  
Ở đây tôi sẽ tập trung vào mã hóa decoder **hifigan**.  

        from Modules.hifigan import Decoder
        decoder = Decoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                resblock_kernel_sizes = args.decoder.resblock_kernel_sizes,
                upsample_rates = args.decoder.upsample_rates,
                upsample_initial_channel=args.decoder.upsample_initial_channel,
                resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                upsample_kernel_sizes=args.decoder.upsample_kernel_sizes) 

Chúng ta cùng đi sâu vào Decoder hifigan



Nó nhận đầu vào từ một mô hình nhận dạng giọng nói tự động (ASR), đường cong F0 (tần số cơ bản), thông tin về độ ồn (N), và một vector phong cách (s).  
Mô hình sử dụng các khối AdainResBlk1d, có thể là để điều chỉnh phong cách của âm thanh đầu ra.  
Nó có các lớp xử lý riêng cho F0 và độ ồn.  
Mô hình sử dụng một Generator, có thể là để tạo ra các đặc trưng âm thanh cụ thể.  
Trong quá trình huấn luyện, nó áp dụng một số kỹ thuật augmentation ngẫu nhiên cho F0 và độ ồn.  
Đầu ra cuối cùng là một biểu diễn âm thanh đã được xử lý, có thể là một dạng spectrogram hoặc các đặc trưng âm thanh khác.  

    text_encoder = TextEncoder(channels=args.hidden_dim, kernel_size=5, depth=args.n_layer, n_symbols=args.n_token)
    
    class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)

        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels),
                actv,
                nn.Dropout(0.2),
            ))
        # self.cnn = nn.Sequential(*self.cnn)

        self.lstm = nn.LSTM(channels, channels // 2, 1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths, m):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.to(input_lengths.device).unsqueeze(1)
        x.masked_fill_(m, 0.0)

        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)

        x = x.transpose(1, 2)  # [B, T, chn]

        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)

        x = x.transpose(-1, -2)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

        x_pad[:, :, :x.shape[-1]] = x
        x = x_pad.to(x.device)

        x.masked_fill_(m, 0.0)

        return x

    def inference(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x

    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask

Mục đích chung:
Lớp này được thiết kế để mã hóa văn bản đầu vào thành các biểu diễn vector có nghĩa. Nó thường được sử dụng trong các mô hình xử lý ngôn ngữ tự nhiên hoặc tổng hợp giọng nói.
Cấu trúc:

Embedding layer: Chuyển đổi các token văn bản thành các vector nhúng.
Các lớp CNN: Xử lý thông tin ngữ cảnh cục bộ.
Lớp LSTM hai chiều: Nắm bắt thông tin ngữ cảnh dài hạn.


Chi tiết từng bước:
a. Khởi tạo (init):

Tạo lớp embedding với số lượng ký hiệu (n_symbols) và số kênh đầu ra.
Xây dựng một chuỗi các lớp CNN với normalization, activation và dropout.
Khởi tạo một lớp LSTM hai chiều.

b. Forward pass:

Embedding: Chuyển đổi đầu vào x thành các vector nhúng.
Xử lý CNN:

Chuyển vị ma trận để phù hợp với đầu vào của CNN.
Áp dụng mặt nạ (mask) để loại bỏ padding.
Áp dụng các lớp CNN, sau mỗi lớp lại áp dụng mask.


Xử lý LSTM:

Chuyển vị ma trận lại.
Sử dụng pack_padded_sequence để xử lý hiệu quả các chuỗi có độ dài khác nhau.
Áp dụng LSTM hai chiều.
Giải nén kết quả bằng pad_packed_sequence.


Xử lý sau cùng:

Chuyển vị ma trận kết quả.
Tạo một tensor mới (x_pad) với kích thước phù hợp và điền kết quả vào.
Áp dụng mask cuối cùng để loại bỏ padding.




Đặc điểm quan trọng:

Sử dụng weight normalization và layer normalization để ổn định quá trình học.
Sử dụng dropout để giảm overfitting.
Xử lý hiệu quả các chuỗi có độ dài khác nhau trong một batch.
Sử dụng masking để đảm bảo rằng padding không ảnh hưởng đến kết quả.


Đầu ra:
Kết quả là một tensor 3D chứa biểu diễn của văn bản đầu vào, với thông tin ngữ cảnh cả cục bộ (từ CNN) và dài hạn (từ LSTM).


    class StyleEncoder(nn.Module):
    def __init__(self, dim_in=48, style_dim=48, max_conv_dim=384):
        super().__init__()
        blocks = []
        blocks += [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]

        repeat_num = 4
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.Linear(dim_out, style_dim)

    def forward(self, x):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        s = self.unshared(h)

        return s

Mô hình StyleEncoder này được thiết kế để mã hóa phong cách (style) từ một đầu vào, thường là một đặc trưng âm thanh như một mel-spectrogram. Đây là một phần quan trọng trong các hệ thống tổng hợp giọng nói hoặc chuyển đổi giọng nói. Hãy phân tích chi tiết:

Mục đích:

Mô hình này nhận đầu vào là một tensor 2D (có thể là mel-spectrogram) và mã hóa nó thành một vector phong cách có kích thước cố định.
Vector phong cách này sau đó có thể được sử dụng để điều khiển các khía cạnh của giọng nói tổng hợp như âm sắc, cảm xúc, hoặc cách phát âm.


Cấu trúc:

Mô hình bắt đầu với một lớp tích chập 2D.
Tiếp theo là một chuỗi các khối ResBlk (Residual Block) với việc giảm kích thước (downsample).
Sau đó là các lớp xử lý cuối cùng bao gồm LeakyReLU, tích chập, pooling trung bình thích ứng.
Cuối cùng, một lớp tuyến tính để tạo ra vector phong cách cuối cùng.


Đặc điểm kỹ thuật:

Sử dụng spectral normalization để ổn định quá trình huấn luyện.
Sử dụng ResBlk để cho phép thông tin flow qua mạng dễ dàng hơn.
Áp dụng LeakyReLU làm hàm kích hoạt để xử lý các giá trị âm.
Sử dụng AdaptiveAvgPool2d để giảm kích thước feature map xuống 1x1, giúp tổng hợp thông tin.


Ứng dụng:

Trong hệ thống tổng hợp giọng nói: Mô hình này có thể được sử dụng để trích xuất phong cách từ một mẫu âm thanh, sau đó áp dụng phong cách đó vào văn bản mới để tạo ra giọng nói có cùng phong cách.
Trong chuyển đổi giọng nói: Có thể được sử dụng để nắm bắt phong cách của giọng nói nguồn và mục tiêu, giúp trong quá trình chuyển đổi.


Đầu ra:

Mô hình trả về một vector s, đại diện cho phong cách đã được mã hóa từ đầu vào.



Tóm lại, StyleEncoder này là một công cụ mạnh mẽ để nắm bắt và mã hóa các đặc trưng phong cách từ dữ liệu âm thanh, đóng vai trò quan trọng trong việc tạo ra hoặc điều chỉnh giọng nói tổng hợp với các đặc điểm phong cách mong muốn.

    class AudioDiffusionModel(Model1d):
        def __init__(self, **kwargs):
            super().__init__(**{**get_default_model_kwargs(), **kwargs})
    
        def sample(self, *args, **kwargs):
            return super().sample(*args, **{**get_default_sampling_kwargs(), **kwargs})


    class AudioDiffusionConditional(Model1d):
        def __init__(
            self,
            embedding_features: int,
            embedding_max_length: int,
            embedding_mask_proba: float = 0.1,
            **kwargs,
        ):
            self.embedding_mask_proba = embedding_mask_proba
            default_kwargs = dict(
                **get_default_model_kwargs(),
                unet_type="cfg",
                context_embedding_features=embedding_features,
                context_embedding_max_length=embedding_max_length,
            )
            super().__init__(**{**default_kwargs, **kwargs})

        def forward(self, *args, **kwargs):
            default_kwargs = dict(embedding_mask_proba=self.embedding_mask_proba)
            return super().forward(*args, **{**default_kwargs, **kwargs})
    
        def sample(self, *args, **kwargs):
            default_kwargs = dict(
                **get_default_sampling_kwargs(),
                embedding_scale=5.0,
            )
            return super().sample(*args, **{**default_kwargs, **kwargs})

Mô hình này là một AudioDiffusionModel, được thiết kế để tạo ra hoặc chỉnh sửa âm thanh sử dụng kỹ thuật khuếch tán (diffusion). Đây là một phương pháp tiên tiến trong lĩnh vực tổng hợp âm thanh. Hãy phân tích chi tiết:

AudioDiffusionModel:

Đây là lớp cơ bản cho mô hình khuếch tán âm thanh.
Nó kế thừa từ Model1d, cho thấy nó làm việc với dữ liệu âm thanh 1 chiều.
Phương thức sample() được sử dụng để tạo ra mẫu âm thanh mới.


AudioDiffusionConditional:

Đây là một phiên bản nâng cao hơn, cho phép điều kiện hóa quá trình tạo âm thanh.
Nó sử dụng một embedding để điều khiển quá trình tạo âm thanh.
Các tham số quan trọng:

embedding_features: Số chiều của vector embedding
embedding_max_length: Độ dài tối đa của chuỗi embedding
embedding_mask_proba: Xác suất che (mask) embedding trong quá trình huấn luyện




Chức năng chính:

Tạo ra âm thanh mới: Mô hình có thể tạo ra âm thanh từ nhiễu ngẫu nhiên.
Chỉnh sửa âm thanh: Có thể được sử dụng để thay đổi âm thanh hiện có.
Điều kiện hóa: AudioDiffusionConditional cho phép điều khiển quá trình tạo âm thanh dựa trên các điều kiện đầu vào (ví dụ: văn bản mô tả, loại âm thanh, v.v.)


Ứng dụng tiềm năng:

Tổng hợp giọng nói: Tạo ra giọng nói mới hoặc sửa đổi giọng nói hiện có.
Tạo nhạc: Có thể được sử dụng để tạo ra các đoạn nhạc mới.
Hiệu ứng âm thanh: Tạo hoặc chỉnh sửa các hiệu ứng âm thanh cho phim, trò chơi, v.v.
Khôi phục âm thanh: Có thể được sử dụng để cải thiện chất lượng của âm thanh bị hư hỏng.


Đặc điểm kỹ thuật:

Sử dụng UNet với Classifier-Free Guidance (CFG) trong phiên bản có điều kiện.
Có khả năng điều chỉnh tỷ lệ embedding (embedding_scale) trong quá trình lấy mẫu.
Sử dụng kỹ thuật masking embedding trong quá trình huấn luyện để tăng tính robust.



Tóm lại, đây là một mô hình mạnh mẽ và linh hoạt cho việc tạo ra và chỉnh sửa âm thanh, với khả năng điều khiển chi tiết thông qua các điều kiện đầu vào. Nó có thể được ứng dụng trong nhiều lĩnh vực liên quan đến xử lý và tổng hợp âm thanh.



    class StyleTransformer1d(nn.Module):
    def __init__(
        self,
        num_layers: int,
        channels: int,
        num_heads: int,
        head_features: int,
        multiplier: int,
        use_context_time: bool = True,
        use_rel_pos: bool = False,
        context_features_multiplier: int = 1,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
        context_features: Optional[int] = None,
        context_embedding_features: Optional[int] = None,
        embedding_max_length: int = 512,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                StyleTransformerBlock(
                    features=channels + context_embedding_features,
                    head_features=head_features,
                    num_heads=num_heads,
                    multiplier=multiplier,
                    style_dim=context_features,
                    use_rel_pos=use_rel_pos,
                    rel_pos_num_buckets=rel_pos_num_buckets,
                    rel_pos_max_distance=rel_pos_max_distance,
                )
                for i in range(num_layers)
            ]
        )

        self.to_out = nn.Sequential(
            Rearrange("b t c -> b c t"),
            nn.Conv1d(
                in_channels=channels + context_embedding_features,
                out_channels=channels,
                kernel_size=1,
            ),
        )
        
        use_context_features = exists(context_features)
        self.use_context_features = use_context_features
        self.use_context_time = use_context_time

        if use_context_time or use_context_features:
            context_mapping_features = channels + context_embedding_features

            self.to_mapping = nn.Sequential(
                nn.Linear(context_mapping_features, context_mapping_features),
                nn.GELU(),
                nn.Linear(context_mapping_features, context_mapping_features),
                nn.GELU(),
            )
        
        if use_context_time:
            assert exists(context_mapping_features)
            self.to_time = nn.Sequential(
                TimePositionalEmbedding(
                    dim=channels, out_features=context_mapping_features
                ),
                nn.GELU(),
            )

        if use_context_features:
            assert exists(context_features) and exists(context_mapping_features)
            self.to_features = nn.Sequential(
                nn.Linear(
                    in_features=context_features, out_features=context_mapping_features
                ),
                nn.GELU(),
            )
            
        self.fixed_embedding = FixedEmbedding(
            max_length=embedding_max_length, features=context_embedding_features
        )
        

    def get_mapping(
        self, time: Optional[Tensor] = None, features: Optional[Tensor] = None
    ) -> Optional[Tensor]:
        """Combines context time features and features into mapping"""
        items, mapping = [], None
        # Compute time features
        if self.use_context_time:
            assert_message = "use_context_time=True but no time features provided"
            assert exists(time), assert_message
            items += [self.to_time(time)]
        # Compute features
        if self.use_context_features:
            assert_message = "context_features exists but no features provided"
            assert exists(features), assert_message
            items += [self.to_features(features)]

        # Compute joint mapping
        if self.use_context_time or self.use_context_features:
            mapping = reduce(torch.stack(items), "n b m -> b m", "sum")
            mapping = self.to_mapping(mapping)

        return mapping
            
    def run(self, x, time, embedding, features):
        
        mapping = self.get_mapping(time, features)
        x = torch.cat([x.expand(-1, embedding.size(1), -1), embedding], axis=-1)
        mapping = mapping.unsqueeze(1).expand(-1, embedding.size(1), -1)
        
        for block in self.blocks:
            x = x + mapping
            x = block(x, features)
        
        x = x.mean(axis=1).unsqueeze(1)
        x = self.to_out(x)
        x = x.transpose(-1, -2)
        
        return x
        
    def forward(self, x: Tensor, 
                time: Tensor, 
                embedding_mask_proba: float = 0.0,
                embedding: Optional[Tensor] = None, 
                features: Optional[Tensor] = None,
               embedding_scale: float = 1.0) -> Tensor:
        
        b, device = embedding.shape[0], embedding.device
        fixed_embedding = self.fixed_embedding(embedding)
        if embedding_mask_proba > 0.0:
            # Randomly mask embedding
            batch_mask = rand_bool(
                shape=(b, 1, 1), proba=embedding_mask_proba, device=device
            )
            embedding = torch.where(batch_mask, fixed_embedding, embedding)

        if embedding_scale != 1.0:
            # Compute both normal and fixed embedding outputs
            out = self.run(x, time, embedding=embedding, features=features)
            out_masked = self.run(x, time, embedding=fixed_embedding, features=features)
            # Scale conditional output using classifier-free guidance
            return out_masked + (out - out_masked) * embedding_scale
        else:
            return self.run(x, time, embedding=embedding, features=features)
        
        return x
Module StyleTransformer1d này được thiết kế để thực hiện việc chuyển đổi phong cách (style transfer) cho dữ liệu âm thanh 1 chiều. Đây là một phần quan trọng trong hệ thống tổng hợp hoặc chỉnh sửa âm thanh có điều kiện. Hãy phân tích các chức năng chính của nó:

Kiến trúc:

Sử dụng nhiều lớp StyleTransformerBlock, cho phép xử lý thông tin phong cách và nội dung một cách hiệu quả.
Có khả năng xử lý thông tin thời gian và đặc trưng bổ sung.


Xử lý ngữ cảnh:

Có thể sử dụng thông tin thời gian (time) và đặc trưng (features) để tạo ra một "mapping" ngữ cảnh.
Sử dụng PositionalEmbedding để mã hóa thông tin vị trí thời gian.


Embedding cố định:

Sử dụng FixedEmbedding để tạo ra một embedding cố định, có thể được sử dụng để so sánh hoặc làm chuẩn.


Điều chỉnh phong cách:

Cho phép điều chỉnh mức độ ảnh hưởng của phong cách thông qua tham số embedding_scale.
Hỗ trợ kỹ thuật classifier-free guidance để kiểm soát mức độ ảnh hưởng của điều kiện.


Masking ngẫu nhiên:

Có khả năng áp dụng masking ngẫu nhiên lên embedding trong quá trình huấn luyện, giúp tăng tính robust của mô hình.


Xử lý đầu vào:

Kết hợp dữ liệu đầu vào (x) với embedding và thông tin ngữ cảnh.
Xử lý thông qua các khối transformer được điều chỉnh.


Đầu ra:

Tổng hợp thông tin từ các time steps và chuyển đổi kích thước để phù hợp với định dạng đầu ra mong muốn.



Tóm lại, StyleTransformer1d này được sử dụng để:

Áp dụng một phong cách cụ thể lên dữ liệu âm thanh đầu vào.
Điều chỉnh mức độ ảnh hưởng của phong cách.
Xử lý thông tin ngữ cảnh (thời gian, đặc trưng) để hướng dẫn quá trình chuyển đổi.
Tạo ra âm thanh mới hoặc chỉnh sửa âm thanh hiện có dựa trên các điều kiện đầu vào.

Module này là một thành phần quan trọng trong các hệ thống tổng hợp âm thanh tiên tiến, cho phép kiểm soát chi tiết về phong cách và nội dung của âm thanh được tạo ra.