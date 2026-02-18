# bevy_autodiff-macros

Procedural macros for [`bevy_autodiff`](https://crates.io/crates/bevy_autodiff).

This crate provides:

- **`#[autodiff]`** — attribute macro that transforms regular Rust functions into dual-use functions generic over `T: DiffNum`. Call with plain floats for direct evaluation, or with `Var` inside `with_context` for AD graph construction.
- **`#[autodiff(stable_derivatives)]`** — automatically routes `pow`/`div` through logarithmic derivative variants for f32-stable second-order derivatives.

## Usage

This crate is not intended to be used directly. Add it via the `proc-macros` feature on `bevy_autodiff`:

```toml
[dependencies]
bevy_autodiff = { version = "0.8", features = ["proc-macros"] }
```

See the [`bevy_autodiff` documentation](https://docs.rs/bevy_autodiff) for usage examples.

## License

MIT
