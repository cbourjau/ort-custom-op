{
  description = "Example Rust development environment for Zero to Nix";

  # Flake inputs
  inputs = {
    nixpkgs.url = "nixpkgs";
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


        (final: prev: {
          onnxruntime = prev.onnxruntime.overrideAttrs (oldAttrs: {
            src = prev.fetchFromGitHub {
              owner = "microsoft";
              repo = "onnxruntime";
              rev = "rel-1.16.0";
              sha256 = "";
              fetchSubmodules = true;
            };
            # Speed up build by skipping upstream tests
            cmakeFlags = prev.lib.filter (x: !(prev.lib.strings.hasPrefix "-DFETCHCONTENT_SOURCE_DIR_EIGEN" x)) (
              prev.lib.lists.subtractLists
                ["-Donnxruntime_BUILD_UNIT_TESTS=ON"]
                prev.onnxruntime.cmakeFlags
            ) ++ ["-Donnxruntime_BUILD_UNIT_TESTS=OFF"];

            buildInputs = prev.onnxruntime.buildInputs ++ [prev.pkgs.eigen];
          });
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
        in
        {
          default = pkgs.mkShell {
            # The Nix packages provided in the environment
            packages = (with pkgs; [
              python3
              python310Packages.flake8
              python310Packages.pytest
              python310Packages.onnx
              # python310Packages.onnxruntime  # build error...
              mypy
              black
              # The package provided by our custom overlay. Includes cargo, Clippy, cargo-fmt,
              # rustdoc, rustfmt, and other tools.
              rustToolchain
            ]) ++ pkgs.lib.optionals pkgs.stdenv.isDarwin (with pkgs; [ libiconv ]);
          };
        });
    };
}
