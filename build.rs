// build.rs derived from https://github.com/oxideai/mlx-rs (MIT/Apache-2.0)

extern crate cmake;

fn main() {
  let mut config = cmake::Config::new("mlx-c");
  config.very_verbose(true);
  config.define("CMAKE_INSTALL_PREFIX", ".");
  config.define("MLX_BUILD_METAL", "ON");
  config.define("MLX_BUILD_ACCELERATE", "ON");
  let dst = config.build();
  println!("cargo:rustc-link-search=native={}/build/lib", dst.display());
  println!("cargo:rustc-link-lib=static=mlx");
  println!("cargo:rustc-link-lib=static=mlxc");
  println!("cargo:rustc-link-lib=dylib=c++");
  println!("cargo:rustc-link-lib=dylib=objc");
  println!("cargo:rustc-link-lib=framework=Foundation");
  println!("cargo:rustc-link-lib=framework=Metal");
  println!("cargo:rustc-link-lib=framework=Accelerate");
}
