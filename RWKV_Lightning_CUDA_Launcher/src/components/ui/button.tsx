import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import type { ComponentProps } from "react";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex shrink-0 items-center justify-center gap-1.5 whitespace-nowrap rounded-lg font-medium transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring/60 disabled:pointer-events-none disabled:opacity-50 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default:
          "border border-primary bg-primary text-primary-foreground hover:opacity-90",
        outline:
          "border border-border bg-background text-foreground hover:bg-muted",
        secondary:
          "border border-transparent bg-secondary text-secondary-foreground hover:bg-accent",
        ghost:
          "border border-transparent text-muted-foreground hover:bg-muted hover:text-foreground",
        destructive:
          "border border-destructive bg-destructive text-destructive-foreground hover:opacity-90",
        danger:
          "border border-border bg-background text-destructive hover:bg-destructive/10",
        link: "border border-transparent text-info underline underline-offset-2 hover:opacity-80",
      },
      size: {
        xs: "h-7 px-2.5 text-xs",
        sm: "h-8 px-3 text-xs",
        default: "h-[34px] px-3.5 text-[13px]",
        lg: "h-9 px-4 text-[13px]",
        icon: "size-7 p-0",
        "icon-sm": "size-8 p-0",
      },
    },
    defaultVariants: { variant: "outline", size: "default" },
  },
);

export interface ButtonProps
  extends ComponentProps<"button">,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

export function Button({
  className,
  variant,
  size,
  asChild = false,
  type,
  ...props
}: ButtonProps) {
  const Component = asChild ? Slot : "button";
  return (
    <Component
      className={cn(buttonVariants({ variant, size }), className)}
      {...(asChild ? {} : { type: type ?? "button" })}
      {...props}
    />
  );
}

export { buttonVariants };
