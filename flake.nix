{
  description = "Example Rust development environment for Zero to Nix";

  # Flake inputs
  inputs = {
    nixpkgs.url = "github:cbourjau/nixpkgs/onnxruntime-macos"; # also valid: "nixpkgs"
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
              owner = "cbourjau";
              repo = "onnxruntime";
              rev = "69102883724a844f476bc4ab60e0aba040d6f823";
              sha256 = "sha256-xRvvkq9EQxj6zHjyOb20qS6Lf5ASHDK6zIXnBGmVR2s=";
              fetchSubmodules = true;
            };
            # Speed up build by skipping upstream tests
            cmakeFlags = (
              prev.lib.lists.subtractLists
                ["-Donnxruntime_BUILD_UNIT_TESTS=ON"]
                prev.onnxruntime.cmakeFlags
            ) ++ ["-Donnxruntime_BUILD_UNIT_TESTS=OFF"];
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
              python310Packages.onnxruntime
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
