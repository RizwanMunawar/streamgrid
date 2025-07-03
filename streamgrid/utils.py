# utils.py
"""Minimal utilities."""

import cv2
import numpy as np


def draw_text_bg(img, text, pos, scale=0.5, color=(255, 255, 255), bg=(0, 0, 0)):
    """Draw text with background."""
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = pos
    
    if bg:
        cv2.rectangle(img, (x-2, y-h-2), (x+w+2, y+2), bg, -1)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)
    return img
