# -*- coding: UTF-8 -*-
# @function :回译调用谷歌翻译，模拟google token访问

import logging as logger
import urllib.parse as parse

import execjs
import requests
import os
from retranslate_constant import language_short_tencent
from text_tools import judge_translate_english
from translate_tencent import translate_tencent_back

class GoogleToken:
    def __init__(self):
        self.ctx = execjs.compile("""
        function TL(a) {
        var k = "";
        var b = 406644;
        var b1 = 3293161072;
        var jd = ".";
        var $b = "+-a^+6";
        var Zb = "+-3^+b+-f";
        for (var e = [], f = 0, g = 0; g < a.length; g++) {
            var m = a.charCodeAt(g);
            128 > m ? e[f++] = m : (2048 > m ? e[f++] = m >> 6 | 192 : (55296 == (m & 64512) && g + 1 < a.length && 56320 == (a.charCodeAt(g + 1) & 64512) ? (m = 65536 + ((m & 1023) << 10) + (a.charCodeAt(++g) & 1023),
            e[f++] = m >> 18 | 240,
            e[f++] = m >> 12 & 63 | 128) : e[f++] = m >> 12 | 224,
            e[f++] = m >> 6 & 63 | 128),
            e[f++] = m & 63 | 128)
        }
        a = b;
        for (f = 0; f < e.length; f++) a += e[f],
        a = RL(a, $b);
        a = RL(a, Zb);
        a ^= b1 || 0;
        0 > a && (a = (a & 2147483647) + 2147483648);
        a %= 1E6;
        return a.toString() + jd + (a ^ b)
    };
    function RL(a, b) {
        var t = "a";
        var Yb = "+";
        for (var c = 0; c < b.length - 2; c += 3) {
            var d = b.charAt(c + 2),
            d = d >= t ? d.charCodeAt(0) - 87 : Number(d),
            d = b.charAt(c + 1) == Yb ? a >>> d: a << d;
            a = b.charAt(c) == Yb ? a + d & 4294967295 : a ^ d
        }
        return a
    }
    """)

    def get_google_token(self, text):
        """
           获取谷歌访问token
        :param text: str, input sentence
        :return:
        """
        return self.ctx.call("TL", text)


def open_url(url):
    """
      新增header，并request访问
    :param url: str, url地址
    :return: str, 目标url地址返回
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36'}
    req = requests.get(url=url, headers=headers)
    # print('req.txt:')
    # print(req.text.encode('gbk', 'ignore').decode('gbk'))
    return req # .content.decode('utf-8')


def max_length(content):
    """
      超过最大长度就不翻译
    :param content: str, need translate
    :return:
    """
    if len(content) > 4891:
        logger.info("翻译文本超过限制！")
        return 4891
    else:
        return None


def translate_result(result):
    """
      删去无关词
    :param result: str
    :return: str
    """
    result_last = ''
    for res in result[0]:
        if res[0]:
            result_last += res[0]
    return result_last


def any_to_any_translate(google_tokn, content, from_='zh-CN', to_='en'):
    """
       自定义选择
    :param content: str, 4891个字， 用户输入
    :param from_: str, original language
    :param to_:   str, target language
    :return: str, result of translate
    """
    max_len = max_length(content)
    if max_len:
        content = content[0:max_len]
    tk = google_tokn.get_google_token(content)
    content = parse.quote(content)
    url = "http://translate.google.cn/translate_a/single?client=t&sl={0}&tl={1}" \
          "&hl=zh-CN&dt=at&dt=bd&dt=ex&dt=ld&dt=md&dt=qca&dt=rw&dt=rm&dt=ss&dt=t&" \
          "ie=UTF-8&oe=UTF-8&source=btn&ssel=3&tsel=3&kc=0&tk={2}&q={3}".format(from_, to_, tk, content)
    result = open_url(url)
    result_json =  result.json()

    res = translate_result(result_json)
    return res


def any_to_any_translate_back(google_tokn, content, from_='zh-CN', to_='en'):
    """
      中英，英中回译
    :param content:str, 4891个字， 用户输入
    :param from_: str, original language
    :param to_:   str, target language
    :return: str, result of translate
    """
    translate_content = any_to_any_translate(google_tokn, content, from_=from_, to_=to_)
    result = any_to_any_translate(google_tokn, translate_content, from_=to_, to_=from_)
    return result

def count_label_num(path, read_filename):
    read_file = open(path + read_filename, 'rb')
    read_file.readline()
    zeros, ones, twos, threes, fours = 0, 0, 0, 0, 0
    for line in read_file:
        # 数据增强准备
        line = line.decode('gbk', 'ignore')
        line_split = line.strip('\n').strip('\r').split('\t')
        if line_split[2].strip() == '0':
            zeros += 1
        elif line_split[2].strip() == '1':
            ones += 1
        elif line_split[2].strip() == '2':
            twos += 1
        elif line_split[2].strip() == '3':
            threes += 1
        elif line_split[2].strip() == '4':
            fours += 1
    read_file.close()
    return [zeros, ones, twos, threes, fours]

def aug_translate(path, read_filename, write_file, class_proportion):
    #使用回译进行数据增强
    read_file = open(path + read_filename, 'rb')
    read_file.readline()

    for line in read_file:
        write_file.write(line)
        line = line.decode('utf-8', 'ignore')
        line_split = line.strip('\n').strip('\r').split('\t')
        for language_short_google_one in language_short_tencent[:class_proportion[int(line_split[-1])] - 1]:
            print("org string is ",line_split[1])
            # text_translate = any_to_any_translate_back(google_tokn, line_split[1], from_='zh', to_=language_short_google_one)
            text_translate = translate_tencent_back(line_split[1], from_='zh', to_=language_short_google_one)
            if text_translate == None or text_translate.strip() == "":
                print("is none!")
                text_translate = line_split[1]
            print("in language : ", language_short_google_one)
            print("translate back string is ",text_translate)
            write_line = line_split[0]+'\t'+text_translate+'\t'+line_split[-1]+'\n'
            write_file.write(write_line.encode('utf-8', 'ignore'))
    read_file.close()



