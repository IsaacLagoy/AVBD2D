import colorsys

def get_color(n: int, tot_colors: int) -> tuple[int, int, int]:
    """
    Return the nth color in a set of evenly spaced hues.

    Args:
        n: index of the color (0 <= n < tot_colors)
        tot_colors: total number of colors

    Returns:
        (r, g, b) tuple with values in [0, 255]
    """
    if tot_colors <= 0:
        raise ValueError("tot_colors must be > 0")
    if not (0 <= n < tot_colors):
        raise ValueError("n must satisfy 0 <= n < tot_colors")

    hue = n / tot_colors  # evenly spaced hue in [0,1)
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)  # full saturation + value
    return (int(r * 255), int(g * 255), int(b * 255))