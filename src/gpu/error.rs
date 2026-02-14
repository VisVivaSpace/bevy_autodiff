//! GPU error types.

use std::fmt;

/// Errors that can occur during GPU operations.
#[derive(Debug)]
#[non_exhaustive]
pub enum GpuError {
    /// No suitable GPU adapter found.
    AdapterRequest(wgpu::RequestAdapterError),
    /// Failed to request a GPU device.
    DeviceRequest(wgpu::RequestDeviceError),
    /// Input array count does not match the compiled graph.
    InputMismatch { expected: usize, got: usize },
    /// Sample count differs between input arrays.
    SampleCountMismatch {
        input_idx: usize,
        expected: usize,
        got: usize,
    },
    /// No samples provided.
    EmptySamples,
    /// GPU buffer mapping failed.
    BufferMap(wgpu::BufferAsyncError),
    /// GPU poll failed.
    PollError(wgpu::PollError),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AdapterRequest(e) => write!(f, "no suitable GPU adapter found: {e}"),
            Self::DeviceRequest(e) => write!(f, "failed to request GPU device: {e}"),
            Self::InputMismatch { expected, got } => {
                write!(f, "expected {expected} input arrays, got {got}")
            }
            Self::SampleCountMismatch {
                input_idx,
                expected,
                got,
            } => write!(
                f,
                "input[{input_idx}] has {got} samples, expected {expected}"
            ),
            Self::EmptySamples => write!(f, "no samples provided"),
            Self::BufferMap(e) => write!(f, "GPU buffer mapping failed: {e}"),
            Self::PollError(e) => write!(f, "GPU poll failed: {e}"),
        }
    }
}

impl std::error::Error for GpuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::AdapterRequest(e) => Some(e),
            Self::DeviceRequest(e) => Some(e),
            Self::BufferMap(e) => Some(e),
            Self::PollError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<wgpu::RequestAdapterError> for GpuError {
    fn from(e: wgpu::RequestAdapterError) -> Self {
        Self::AdapterRequest(e)
    }
}

impl From<wgpu::RequestDeviceError> for GpuError {
    fn from(e: wgpu::RequestDeviceError) -> Self {
        Self::DeviceRequest(e)
    }
}

impl From<wgpu::BufferAsyncError> for GpuError {
    fn from(e: wgpu::BufferAsyncError) -> Self {
        Self::BufferMap(e)
    }
}

impl From<wgpu::PollError> for GpuError {
    fn from(e: wgpu::PollError) -> Self {
        Self::PollError(e)
    }
}
