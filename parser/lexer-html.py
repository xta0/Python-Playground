# Python LA library
import ply.lex as lex
import re 

tokens = (
    'LANGLE', # <
    'LANGLESLASH', #</
    'RANGLE', #>
    'EQUAL', #=
    'STRING', #"hello"
    'WORD' #welcome!
)
t_ignore = ' ' #shortcut for whitespaces

# ignore state
states = (
    ('htmlcomment','exclusive'),
)
def t_htmlcomment(token):
    r'<!--'
    token.lexer.begin('htmlcomment')

def t_htmlcomment_end(token):
    r'-->'
    token.lexer.lineno += token.value.count('\n')
    token.lexer.begin('INITIAL')


# HTML Tokens
def t_LANGLESLASH(token):
    r'</'
    return token

def t_LANGLE(token):
    r'<'
    return token


def t_RANGLE(token):
    r'>'
    return token

def t_EQUAL(token):
    r'='
    return token

# math double quote string <a href = "abc" />
def t_STRING(token):
    r'"[^"]+"'
    token.value = token.value[1:-1] #remove ""
    return token

def t_WORD(token):
    r'[^ <>\n]+'
    return token     

def t_NUMBER(token):
    r'[0-9]+'
    token.value = int(token.value)
    return token

def t_htmlcomment_error(token):
    token.lexer.skip(1)

def t_newline(token):
    r'\n'
    token.lexer.lineno += 1
    pass

webpage =  '''This is <!-- comment --> 
<b>my</b> webpage!'''
# webpage = """This is 
# <b>my</b> webpage!"""
# webpage = 'hello <!-- comment --> all'
htmllexer = lex.lex()
htmllexer.input(webpage)
while True:
        tok = htmllexer.token() #return next token
        if not tok: 
            break
        print(tok)



