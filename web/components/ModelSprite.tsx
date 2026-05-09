/* eslint-disable @next/next/no-img-element */
import { getModelSprite } from "@/lib/model-sprites";

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH || "";

type ModelSpriteProps = {
  className?: string;
  model?: string | null;
  title?: string;
};

export default function ModelSprite({ className = "", model, title }: ModelSpriteProps) {
  const sprite = getModelSprite(model) || getModelSprite("ensemble");
  if (!sprite) return null;

  return (
    <span className={className} title={title || sprite.name}>
      <img src={`${BASE_PATH}${sprite.image}`} alt="" aria-hidden="true" />
    </span>
  );
}
