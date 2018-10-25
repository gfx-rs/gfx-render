# gfx-render

This crates adds basic functionality around [`gfx-hal`](https://github.com/gfx-rs/gfx) through two primary types `Factory` and `Renderer` that can be instantiated together by `init` function.

`Factory`'s functionality:
1. Allocating wrapped buffers and images using [`gfx-memory`'s](https://github.com/gfx-rs/gfx-memory) `SmartAllocator`.
  Wrapper will release resource automatically on drop.
  Also supports manual deallocation with less overhead than automatic.
1. Preserving deallocated buffers and images until they are not references by GPU's in-progress commands.
  Simply by waiting for all jobs that were recording or in-progress at the moment of deallocation to complete.
1. Uploading data to buffers and images with method chosen based on memory properties.
1. Report features and limits of physical device and capabilities and formats for the surface.
1. Substitute `B::Device` in generic code. `Factory<B>` implements `Device<B>`.

`Renderer`'s functionality:
1. Creating rendering targets. Currently only surfaces from `winit::Window`. But headless mode is planned.
1. Instantiating custom `Render` implementation for each target. Designed to be compatible with [`xfg` crate](https://github.com/omni-viral/xfg-rs) but not limiting to.
1. Kicking off rendering jobs and managing completion through fences.

## License

[license]: #license

This repository is licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributions

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

## What else

Other crates that aim to simplify usage of [`gfx-hal`](https://github.com/gfx-rs/gfx):
* [`gfx-memory`](https://github.com/gfx-rs/gfx-memory) - memory allocators. Used in `gfx-render` internally but `Factory` can give access to the underlying allocator.
* [`gfx-chain`](https://github.com/omni-viral/gfx-chain) - automatic synchronization. Requires up front dependencies declaration.
* [`gfx-mesh`](https://github.com/omni-viral/gfx-mesh) - create meshes from vertex and index data with easy-to-use API. Dependes on `gfx-render`.

