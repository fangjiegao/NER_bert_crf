# coding=utf-8
"""
参数解析
illool@163.com
QQ:122018919
"""
from configparser import ConfigParser

__config = None


def get_config(config_file_path='./config.conf'):
    """
    单例配置获取
    """
    global __config
    if not __config:
        config = ConfigParser()
        config.read(config_file_path)
    else:
        config = __config
    return config
