{
  description = "Example Rust development environment for Zero to Nix";

  # Flake inputs
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs"; # also valid: "nixpkgs"
    rust-overlay.url = "github:oxalica/rust-overlay"; # A helper for Rust + Nix
  };

  # Flake outputs
  outputs = { self, nixpkgs, rust-overlay }:
    let
      # Overlays enable you to customize the Nixpkgs attribute set
      overlays = [
        # Makes a `rust-bin` attribute available in Nixpkgs
        (import rust-overlay)
        # Provides a `rustToolchain` attribute for Nixpkgs that we can use to
        # create a Rust environment
        (self: super: {
          rustToolchain = super.rust-bin.stable.latest.default;
        })
      ];

      # Systems supported
      allSystems = [
        "x86_64-linux" # 64-bit Intel/AMD Linux
        "aarch64-linux" # 64-bit ARM Linux
        "x86_64-darwin" # 64-bit Intel macOS
        "aarch64-darwin" # 64-bit ARM macOS
      ];

      # Helper to provide system-specific attributes
      forAllSystems = f: nixpkgs.lib.genAttrs allSystems (system: f {
        pkgs = import nixpkgs { inherit overlays system; };
      });
    in
    {
      # Development environment output
      devShells = forAllSystems ({ pkgs }:
        let
          lib-path = with pkgs; lib.makeLibraryPath [
            libffi
            openssl
            stdenv.cc.cc
          ];
        in
        {
          default = pkgs.mkShell {
            # The Nix packages provided in the environment
            packages = (with pkgs; [
              python3
              python310Packages.flake8
              # python310Packages.pytest
              # python310Packages.onnx
              # python310Packages.onnxruntime  # build error...
              python310Packages.black
              mypy
              black
              # The package provided by our custom overlay. Includes cargo, Clippy, cargo-fmt,
              # rustdoc, rustfmt, and other tools.
              rustToolchain
            ]) ++ pkgs.lib.optionals pkgs.stdenv.isDarwin (with pkgs; [ libiconv ]);
            # shellHook inspired by:
            # https://gist.github.com/cdepillabout/f7dbe65b73e1b5e70b7baa473dafddb3
            shellHook = ''
# Set LD_LIBRARY_PATH (Linux) for libraries not vendored by pip
export "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${lib-path}"

VENV=.venv
if test ! -d $VENV; then
    python -m venv $VENV
    source ./$VENV/bin/activate
    pip install onnx==1.15 onnxruntime pytest
else
    source ./$VENV/bin/activate
fi
''
            ;
          };
        });
    };
}
