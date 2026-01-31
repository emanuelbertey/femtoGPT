use super::Function;
use crate::tensor::*;

#[cfg(feature = "gpu")]
use super::{gpu, GpuFunction, TensorId};

#[derive(Debug, Clone)]
pub struct RoPE {}

impl RoPE {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}

impl Function for RoPE {
    fn run(
        &mut self,
        inps: &[&GeneralTensor],
        _training: bool,
    ) -> Result<Tensor<f32>, TensorError> {
        let x = inps[0].as_float()?;
        x.map(2, |t| {
            let (num_tokens, head_size) = (t.shape()[0], t.shape()[1]);
            let mut out = t.blob().to_vec();
            for m in 0..num_tokens {
                for i in 0..(head_size / 2) {
                    let theta = (m as f32) / 10000.0f32.powf((2 * i) as f32 / head_size as f32);
                    let cos = theta.cos();
                    let sin = theta.sin();

                    let x0 = t.blob()[m * head_size + i];
                    let x1 = t.blob()[m * head_size + i + head_size / 2];

                    out[m * head_size + i] = x0 * cos - x1 * sin;
                    out[m * head_size + i + head_size / 2] = x0 * sin + x1 * cos;
                }
            }
            Ok(Tensor::raw(&[num_tokens, head_size], out)?)
        })
    }
    fn grad(
        &self,
        _inps: &[&GeneralTensor],
        out_grad: &Tensor<f32>,
    ) -> Result<Vec<Tensor<f32>>, TensorError> {
        let grad = out_grad.map(2, |t| {
            let (num_tokens, head_size) = (t.shape()[0], t.shape()[1]);
            let mut out = t.blob().to_vec();
            for m in 0..num_tokens {
                for i in 0..(head_size / 2) {
                    let theta = (m as f32) / 10000.0f32.powf((2 * i) as f32 / head_size as f32);
                    let cos = theta.cos();
                    let sin = theta.sin();

                    let g0 = t.blob()[m * head_size + i];
                    let g1 = t.blob()[m * head_size + i + head_size / 2];

                    out[m * head_size + i] = g0 * cos + g1 * sin;
                    out[m * head_size + i + head_size / 2] = -g0 * sin + g1 * cos;
                }
            }
            Ok(Tensor::raw(&[num_tokens, head_size], out)?)
        })?;
        Ok(vec![grad])
    }
    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }

    #[cfg(feature = "gpu")]
    fn gpu_impl(&self, out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
        gpu::rope::gpu_impl(out_id, inps)
    }
}
