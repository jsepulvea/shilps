import logging as loggg


loggg.basicConfig(
    level=loggg.DEBUG,
    format='%(levelname)s:%(message)s'
)

class CustomFormatter(loggg.Formatter):
    def format(self, record):
        # Pad the level name to a fixed length, in this case, 5 characters
        levelname_padded = record.levelname.ljust(5)
        record.levelname = levelname_padded
        return super().format(record)
    

# Create the custom formatter
formatter = CustomFormatter('%(levelname)s: %(message)s')

# Get the root logger
logging = loggg.getLogger()

# Set the custom formatter for all handlers of the root logger
for handler in logging.handlers:
    handler.setFormatter(formatter)

