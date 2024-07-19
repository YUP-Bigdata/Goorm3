import re


###################################################################
'''
regular expression
< regular >
input: data
output: data in regular expression
'''
###################################################################


def regular(data):
    return [' '.join(re.sub('[-=+,#/$\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', '', sent.replace('_num_', ' ').replace('\n', ' ')).split()) for sent in data]