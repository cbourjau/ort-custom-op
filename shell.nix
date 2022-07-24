# Taken from https://nixos.wiki/wiki/Rust#Installation_via_rustup
# and https://discourse.nixos.org/t/compile-a-rust-binary-on-macos-dbcrossbar/8612
{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell rec {
  buildInputs = with pkgs; [
    clang
    llvmPackages_latest.llvm
    llvmPackages_latest.bintools
    llvmPackages_latest.lld
    darwin.apple_sdk.frameworks.Security
    lldb
    python3
    python310Packages.flake8
    mypy
    black
    libiconv
  ];
  # https://github.com/rust-lang/rust-bindgen#environment-variables
  LIBCLANG_PATH= pkgs.lib.makeLibraryPath [ pkgs.llvmPackages_latest.libclang.lib ];
  HISTFILE=toString ./.history;
  shellHook = ''
    source .venv/bin/activate
  '';
}
