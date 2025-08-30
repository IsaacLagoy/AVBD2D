import colorsys

def get_color(n: int, tot_colors: int) -> tuple[int, int, int]:
    hue = n / tot_colors  # evenly spaced hue
    # pastel settings: lower saturation, higher value
    sat = 0.4
    val = 0.9
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (int(r * 255), int(g * 255), int(b * 255))
