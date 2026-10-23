# Home-manager module for Voo speech-to-text
#
# Provides a systemd user service for autostart.
# Usage: imports = [ handy.homeManagerModules.default ];
#        services.voo.enable = true;
{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.services.voo;
in
{
  options.services.voo = {
    enable = lib.mkEnableOption "Voo speech-to-text user service";

    package = lib.mkOption {
      type = lib.types.package;
      defaultText = lib.literalExpression "voo.packages.\${system}.voo";
      description = "The Voo package to use.";
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.user.services.voo = {
      Unit = {
        Description = "Voo speech-to-text";
        After = [ "graphical-session.target" ];
        PartOf = [ "graphical-session.target" ];
      };
      Service = {
        ExecStart = "${cfg.package}/bin/voo";
        Restart = "on-failure";
        RestartSec = 5;
      };
      Install.WantedBy = [ "graphical-session.target" ];
    };
  };
}
