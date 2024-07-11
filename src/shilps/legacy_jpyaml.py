import re
from datetime import timedelta
import yaml


# Config YAML parser
PATTERN_TIMEDELTA = re.compile('[0-9]+m[0-9]+s')


def timedelta_representer(dumper, data: timedelta):
    total_secs = int(data.total_seconds())
    minutes = total_secs // 60
    seconds = total_secs % 60
    return dumper.represent_scalar('!dt', '{}m{}s'.format(minutes, seconds))


def timedelta_constructor(loader, node):
    value = loader.construct_scalar(node)[:-1]
    minutes, seconds = map(int, value.split('m'))
    return timedelta(minutes=minutes, seconds=seconds)


yaml.add_representer(timedelta, timedelta_representer)
yaml.add_constructor(u'!dt', timedelta_constructor)
yaml.add_implicit_resolver(u'!dt', PATTERN_TIMEDELTA)
