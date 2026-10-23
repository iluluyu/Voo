import React from "react";

const HandyTextLogo = ({
  width,
  height,
  className,
}: {
  width?: number;
  height?: number;
  className?: string;
}) => {
  return (
    <svg
      width={width}
      height={height}
      className={className}
      viewBox="0 0 930 328"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      {/* V */}
      <path
        d="M40 80 L150 80 L250 215 L350 80 L460 80 L300 280 L200 280 Z"
        className="logo-primary"
      />
      {/* o */}
      <path
        fillRule="evenodd"
        clipRule="evenodd"
        d="M490 180 a100 100 0 1 0 200 0 a100 100 0 1 0 -200 0 Z M540 180 a50 50 0 1 1 100 0 a50 50 0 1 1 -100 0 Z"
        className="logo-primary"
      />
      {/* o */}
      <path
        fillRule="evenodd"
        clipRule="evenodd"
        d="M700 180 a100 100 0 1 0 200 0 a100 100 0 1 0 -200 0 Z M750 180 a50 50 0 1 1 100 0 a50 50 0 1 1 -100 0 Z"
        className="logo-stroke"
      />
    </svg>
  );
};

export default HandyTextLogo;
