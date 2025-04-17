from django import template

register = template.Library()

@register.filter
def abs_value(value):
    """Returns the absolute value of a number."""
    try:
        return abs(float(value))
    except (ValueError, TypeError):
        return value

@register.filter
def mul(value, arg):
    """Multiplies the value by the argument"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return value

@register.filter
def floatformat(value, arg=-1):
    """Formats a float to a specified number of decimal places."""
    try:
        value = float(value)
        arg = int(arg)
        if arg < 0:
            # Default Django behavior for negative precision
            arg = abs(arg)
            if value == int(value):
                return str(int(value))
        return ('{0:.%df}' % arg).format(value)
    except (ValueError, TypeError):
        return value
