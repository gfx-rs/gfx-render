
extern crate crossbeam_channel;
extern crate gfx_hal as hal;
extern crate gfx_memory as mem;
#[macro_use] extern crate error_chain;
#[macro_use] extern crate log;
extern crate winit;

#[cfg(not(any(feature = "gfx-backend-vulkan", feature = "gfx-backend-dx12", feature = "gfx-backend-metal")))]
pub extern crate gfx_backend_empty as empty;

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

pub use init::init;
pub use factory::{Factory, Item, Buffer, Image};
pub use renderer::{Renderer, TargetId};

error_chain!{}
