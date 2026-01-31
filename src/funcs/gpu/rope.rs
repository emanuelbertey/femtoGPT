use super::GpuFunction;
use super::KernelCall;
use crate::graph::TensorId;

pub fn gpu_impl(out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
    let works = inps[0][..inps[0].len() - 2].iter().fold(1, |a, b| a * b);
    let n = inps[0][inps[0].len() - 2];
    let head_size = inps[0][inps[0].len() - 1];

    let forward_source_code = format!(
        "__kernel void calc_{out_id}(
                        __global float* out,
                        __global float* a) {{
        uint id = get_global_id(0);
        if(id < {works}) {{
            for(uint m = 0; m < {n}; m++) {{
                for(uint i = 0; i < {head_size} / 2; i++) {{
                    float theta = (float)m / pow(10000.0f, (float)(2 * i) / (float){head_size});
                    float cos_theta = cos(theta);
                    float sin_theta = sin(theta);
                    
                    uint idx0 = id * {n} * {head_size} + m * {head_size} + i;
                    uint idx1 = id * {n} * {head_size} + m * {head_size} + i + {head_size} / 2;
                    
                    float x0 = a[idx0];
                    float x1 = a[idx1];
                    
                    out[idx0] = x0 * cos_theta - x1 * sin_theta;
                    out[idx1] = x0 * sin_theta + x1 * cos_theta;
                }}
            }}
        }}
    }}"
    );

    let backward_source_code = format!(
        "__kernel void grad_{out_id}(
                        __global float* out,
                        __global float* out_grad,
                        __global float* a,
                        __global float* a_grad) {{
        uint id = get_global_id(0);
        if(id < {works}) {{
            for(uint m = 0; m < {n}; m++) {{
                for(uint i = 0; i < {head_size} / 2; i++) {{
                    float theta = (float)m / pow(10000.0f, (float)(2 * i) / (float){head_size});
                    float cos_theta = cos(theta);
                    float sin_theta = sin(theta);
                    
                    uint idx0 = id * {n} * {head_size} + m * {head_size} + i;
                    uint idx1 = id * {n} * {head_size} + m * {head_size} + i + {head_size} / 2;
                    
                    float g0 = out_grad[idx0];
                    float g1 = out_grad[idx1];
                    
                    a_grad[idx0] += g0 * cos_theta + g1 * sin_theta;
                    a_grad[idx1] += -g0 * sin_theta + g1 * cos_theta;
                }}
            }}
        }}
    }}"
    );

    GpuFunction {
        shared_buffers: vec![],
        forward_funcs: vec![KernelCall {
            source_code: forward_source_code,
            kernel_name: format!("calc_{}", out_id),
            local_work_size: 32,
            global_work_size: works,
        }],
        backward_funcs: vec![KernelCall {
            source_code: backward_source_code,
            kernel_name: format!("grad_{}", out_id),
            local_work_size: 32,
            global_work_size: works,
        }],
    }
}
