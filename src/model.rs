use burn::{
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear,
        LinearConfig, Relu,
    },
    prelude::*,
    tensor::{Bool, Tensor, activation::softmax, backend::Backend},
};

#[derive(Module, Debug)]
pub struct AttentionHead<B: Backend> {
    tril_buffer: Tensor<B, 2, Int>,
    key: Linear<B>,
    query: Linear<B>,
    head_size: usize,
    value: Linear<B>,
    d_input: usize,
    dropout: Dropout,
}

impl<B: Backend> AttentionHead<B> {
    pub fn new(d_input: usize, head_size: usize, block_size: usize, device: &B::Device) -> Self {
        Self {
            tril_buffer: Tensor::ones([block_size, block_size], device).tril(0),
            key: LinearConfig::new(d_input, head_size)
                .with_bias(false)
                .init(device),
            query: LinearConfig::new(d_input, head_size)
                .with_bias(false)
                .init(device),
            value: LinearConfig::new(d_input, head_size)
                .with_bias(false)
                .init(device),
            head_size,
            d_input,
            dropout: DropoutConfig::new(0.2).init(),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>, // [B, T, d_input]
    ) -> Tensor<B, 3> {
        let shape = x.shape();
        let b = shape.dims[0];
        let t = shape.dims[1];
        let k = self.key.forward(x.clone()); // [B, T, head_size]
        let q = self.query.forward(x.clone()); // [B, T, head_size]

        // Compute attention weights
        let k_t = k.transpose(); // [B, head_size, T]
        let mut wei = q.matmul(k_t); // [B, T, T]
        wei = wei.div_scalar(f32::sqrt(self.head_size as f32));

        // Causal mask: expand mask to [B, T, T]
        let mask: Tensor<B, 2, Bool> = self.tril_buffer.clone().equal_elem(0);
        let mask = mask.unsqueeze::<2>().expand([b, t, t]); // [B, T, T]
        wei = wei.mask_fill(mask, f32::NEG_INFINITY);
        wei = softmax(wei, 2); // softmax over last dim

        let v = self.value.forward(x.clone());
        let attn_out = wei.matmul(v); // [B, T, d_input]
        
        self.dropout.forward(attn_out)
    }
}

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    attention_heads: [AttentionHead<B>; 4],
    proj: Linear<B>,
    forward: Linear<B>,
    proj2: Linear<B>,
    layer_norm1: LayerNorm<B>,
    layer_norm2: LayerNorm<B>,
    relu: Relu,
    dropout: Dropout,
}

impl<B: Backend> Block<B> {
    pub fn new(n_embd: usize, n_head: usize, block_size: usize, device: &B::Device) -> Self {
        let head_size = n_embd / n_head;
        Self {
            attention_heads: [
                AttentionHead::new(n_embd, head_size, block_size, device),
                AttentionHead::new(n_embd, head_size, block_size, device),
                AttentionHead::new(n_embd, head_size, block_size, device),
                AttentionHead::new(n_embd, head_size, block_size, device),
            ],
            proj: LinearConfig::new(n_embd, n_embd).init(device),
            forward: LinearConfig::new(n_embd, 4 * n_embd).init(device),
            proj2: LinearConfig::new(4 * n_embd, n_embd).init(device),
            layer_norm1: LayerNormConfig::new(n_embd).init(device),
            layer_norm2: LayerNormConfig::new(n_embd).init(device),
            relu: Relu::new(),
            dropout: DropoutConfig::new(0.2).init(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Pre-attention layer norm
        let x_norm1 = self.layer_norm1.forward(x.clone());
        let mut attn_outs = Vec::new();
        for head in self.attention_heads.iter() {
            let attn_out = head.forward(x_norm1.clone()); // [B, T, d_input]
            attn_outs.push(attn_out); // [B, T, d_input]
        }
        let attn_out = Tensor::cat(attn_outs, 2); // [B, T, n_embd]
        // Projection
        let attn_out = self.proj.forward(attn_out);
        // Residual connection
        let attn_out = attn_out + x.clone(); // [B, T, n_embd]
        // Pre-FFN layer norm
        let attn_out_norm2 = self.layer_norm2.forward(attn_out.clone());
        // Feed forward
        let ff_out = self.forward.forward(attn_out_norm2); // [B, T, 4*n_embd]
        let ff_out = self.relu.forward(ff_out); // [B, T, 4*n_embd]
        // Projection
        let ff_out = self.proj2.forward(ff_out); // [B, T, n_embd]
        // Residual connection
        let out = ff_out + attn_out; // [B, T, n_embd]
        
        self.dropout.forward(out)
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    embedding_table: Embedding<B>,
    position_embedding: Embedding<B>,
    lm_head: Linear<B>,
    blocks: [Block<B>; 3],
    layer_norm: LayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let device = input.device();
        let shape = input.shape();
        let t = shape.dims[1];
        let tok_emb = self.embedding_table.forward(input); // [B, T, n_embd]
        let pos = Tensor::<B, 1, Int>::arange(0..t as i64, &device).unsqueeze::<2>(); // [1, T]
        let pos_emb = self.position_embedding.forward(pos); // [1, T, n_embd]
        let x: Tensor<B, 3> = tok_emb + pos_emb; // [B, T, n_embd]
        let mut x = x;
        for block in self.blocks.iter() {
            x = block.forward(x); // [B, T, n_embd]
        }
        let x = self.layer_norm.forward(x); // [B, T, n_embd]
        let logits = self.lm_head.forward(x); // [B, T, vocab_length]
        let shape = logits.shape(); // B, T, C
        let out = logits.reshape([shape.dims[0] * shape.dims[1], shape.dims[2]]);
        
        self.dropout.forward(out)
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    pub vocab_length: usize,
    pub block_size: usize,
    pub n_embd: usize,
    pub head_size: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            embedding_table: EmbeddingConfig::new(self.vocab_length, self.n_embd).init(device),
            position_embedding: EmbeddingConfig::new(self.block_size, self.n_embd).init(device),
            lm_head: LinearConfig::new(self.n_embd, self.vocab_length).init(device),
            blocks: [
                Block::new(self.n_embd, 4, self.block_size, device),
                Block::new(self.n_embd, 4, self.block_size, device),
                Block::new(self.n_embd, 4, self.block_size, device),
            ],
            layer_norm: LayerNormConfig::new(self.n_embd).init(device),
            dropout: DropoutConfig::new(0.2).init(),
        }
    }
}
