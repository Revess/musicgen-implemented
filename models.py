from musiclm_pytorch import MuLaN, AudioSpectrogramTransformer, TextTransformer, MuLaNEmbedQuantizer
from audiolm_pytorch import HubertWithKmeans, SemanticTransformer, SoundStream, FineTransformer, CoarseTransformer

def build_mulan():
    audio_transformer = AudioSpectrogramTransformer(
        dim = 512,
        depth = 6,
        heads = 8,
        dim_head = 64,
        spec_n_fft = 128,
        spec_win_length = 24,
        spec_aug_stretch_factor = 0.8
    )

    text_transformer = TextTransformer(
        dim = 512,
        depth = 6,
        heads = 8,
        dim_head = 64
    )

    mulan = MuLaN(
        audio_transformer = audio_transformer,
        text_transformer = text_transformer
    )
    return mulan, audio_transformer, text_transformer

def build_wav2vec():
    wav2vec = HubertWithKmeans(
        checkpoint_path = './models/hubert/hubert_base_ls960.pt',
        kmeans_path = './models/hubert/hubert_base_ls960_L9_km500.bin'
    )
    return wav2vec

def build_quantizer(mulan):
    quantizer = MuLaNEmbedQuantizer(
        mulan = mulan,                          # pass in trained mulan from above
        conditioning_dims = (1024, 1024, 1024), # say all three transformers have model dimensions of 1024
        namespaces = ('semantic', 'coarse', 'fine')
    )

    return quantizer

def build_semantic_transformer(quantizer, wav2vec):
    semantic_transformer = SemanticTransformer(
        num_semantic_tokens = wav2vec.codebook_size,
        dim = 1024,
        depth = 6,
        audio_text_condition = True      # this must be set to True (same for CoarseTransformer and FineTransformers)
    )
    return semantic_transformer

def build_sound_stream():
    soundstream = SoundStream(
        codebook_size = 4096,
        rq_num_quantizers = 8,
        rq_groups = 2,                       # this paper proposes using multi-headed residual vector quantization - https://arxiv.org/abs/2305.02765
        use_lookup_free_quantizer = True,    # whether to use residual lookup free quantization - there are now reports of successful usage of this unpublished technique
        use_finite_scalar_quantizer = False, # whether to use residual finite scalar quantization
        attn_window_size = 128,              # local attention receptive field at bottleneck
        attn_depth = 2                       # 2 local attention transformer blocks - the soundstream folks were not experts with attention, so i took the liberty to add some. encodec went with lstms, but attention should be better
    )
    return soundstream

def build_coarse_transformer(wav2vec):
    coarse_transformer = CoarseTransformer(
        num_semantic_tokens = wav2vec.codebook_size,
        codebook_size = 1024*4,
        num_coarse_quantizers = 3,
        dim = 1024,
        depth = 6,
        flash_attn = True,
        audio_text_condition = True
    )
    return coarse_transformer

def build_fine_transformer():
    fine_transformer = FineTransformer(
        num_coarse_quantizers = 3,
        num_fine_quantizers = 5,
        codebook_size = 1024*4,
        dim = 1024,
        depth = 6,
        flash_attn = True,
        audio_text_condition = True
    )
    return fine_transformer