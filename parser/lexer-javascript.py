# Python LA library
import ply.lex as lex
import re 


# Javascript tokens
def t_identifier(token):
    r'[a-zA-Z][\w]*'
    return token

def t_numbers(token):
    r'-?[0-9]+(?:\.[0-9]+)?'
    token.value = float(token.value)
    return token

def t_commnet(token):
    r'//[^\n]*'
    pass
