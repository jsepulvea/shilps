import platform

if platform.system() in ['Linux', 'Darwin']:  # Darwin is macOS
    SUCCESS_STRING = " \U0001F40D \u2B50"  # Snake and star emoji
elif platform.system() == 'Windows':
    SUCCESS_STRING = ""

