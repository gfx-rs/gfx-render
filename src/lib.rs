
extern crate crossbeam_channel;
#[macro_use]
extern crate failure;
extern crate gfx_hal as hal;
extern crate gfx_memory as mem;
#[macro_use]
extern crate log;
extern crate winit;

#[cfg(feature = "profile")]
extern crate flame;

#[cfg(feature = "profile")]
macro_rules! profile {
    ($name:tt) => {
        let guard = ::flame::start_guard(concat!("'", $name, "' at : ", line!()));
    }
}

#[cfg(not(feature = "profile"))]
macro_rules! profile {
    ($name:tt) => {}
}

#[cfg(feature = "gfx-backend-vulkan")]
pub extern crate gfx_backend_vulkan as vulkan;

#[cfg(feature = "gfx-backend-dx12")]
pub extern crate gfx_backend_dx12 as dx12;

#[cfg(feature = "gfx-backend-metal")]
pub extern crate gfx_backend_metal as metal;

mod backend;
mod escape;
mod factory;
mod reclamation;
mod renderer;
mod upload;
mod init;

pub use backend::BackendEx;
pub use init::init;
pub use factory::{Buffer, Factory, Image, Item};
pub use renderer::{Render, Renderer, TargetId};
