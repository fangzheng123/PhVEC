#encoding: utf-8
'''
@File    :   test.py
@Time    :   2021/04/20 11:27:14
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''


from pypinyin import lazy_pinyin, pinyin, Style


if __name__ == "__main__":
    print(pinyin('中', style=Style.INITIALS)[0][0])
    print(pinyin('中', style=Style.FINALS)[0][0])

    
    

