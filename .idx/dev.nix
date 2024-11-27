# To learn more about how to use Nix to configure your environment
# see: https://developers.google.com/idx/guides/customize-idx-env
{ pkgs, ... }: {
  # Which nixpkgs channel to use.
  channel = "stable-24.05"; # or "unstable"

  # Use https://search.nixos.org/packages to find packages
  packages = [
    # pkgs.go
    pkgs.python311
    pkgs.python311Packages.pip
    # pkgs.nodejs_20
    # pkgs.nodePackages.nodemon
  ];

  # Sets environment variables in the workspace
  env = {
    PORT=5000;
  };
  idx = {
    extensions = [ "ms-python.python" ];

    # Enable previews
    previews = {
      enable = true;
      previews = {
        web = {
          command = ["./devserver.sh"];
          env = { PORT = "$PORT"; };
          manager = "web";
        };
      };
    };


    # Workspace lifecycle hooks
    workspace = {
      # Runs when a workspace is first created
      onCreate = {
        install =
          "python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt";
        default.openFiles = ["run.py" ];
      };
      # Runs when the workspace is (re)started
    };
  };
}
