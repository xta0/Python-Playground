
def get_row(matrix, row):
    return matrix[row]

def get_column(matrix, column_number):
    column = []
    h = len(matrix)
    for i in range(h):
        column.append(matrix[i][column_number])
    return column

def matrix_addition(matrixA, matrixB):

    matrixSum = []
    row = []
    h = len(matrixA)
    w = len(matrixA[0])
    for i in range(h):
        for j in range(w):
            row.append(matrixA[i][j] + matrixB[i][j])
        matrixSum.append(row)
        row = []
    
    return matrixSum
    
def dot_product(vector_one, vector_two):
    sum = 0;
    for i in range(len(vector_one)):
        sum += vector_one[i] * vector_two[i]
    return sum

def matrix_multiplication(matrixA, matrixB):
    
    m_rows = len(matrixA) #A: mxn
    p_columns = len(matrixB[0]) #B: nxp
        
    result = []

    for i in range(m_rows):
        for j in range(p_columns):
            row = get_row(matrixA,i)
            column = get_column(matrixB,j)
            val = dot_product(row,column)
            row_result.append(val)
        result.append(row_result)
        row_result = []
    
    
    return result


def transpose(matrix):
    matrix_transpose = []
    for j in range(len(matrix[0])):
        vec = []
        for i in range(len(matrix)):
            vec.append(matrix[i][j])
        matrix_transpose.append(vec)
    
    return matrix_transpose


# multiplication using transpose
def matrix_multiplication_2(matrixA, matrixB):
    product = []
    matrixBtrans = transpose(matrixB)
    rowsA = len(matrixA)
    rowsB = len(matrixBtrans)
    rows_result = []
    for i in range(rowsA):
        for j in range(rowsB):
            val = dot_product(matrixA[i], matrixBtrans[j])
            rows_result.append(val)
        product.append(rows_result)
        rows_result = []

    return product

# identity matrix
def identity_matrix(n):
    
    identity = []
    
    for i in range (n):
        vec = []
        for j in range(n):
            if i==j:
                vec.append(1)
            else:
                vec.append(0)
        identity.append(vec)
        vec = []
    return identity


def inverse_matrix(matrix):
    
    inverse = []
    
    if len(matrix) != len(matrix[0]):
        raise ValueError('The matrix must be square')
    
    if (len(matrix) > 2):
        raise ValueError('The matrix size must less than 2x2')
    

    if len(matrix)==1 : 
        vec = []
        vec.append(1.0/matrix[0][0])
        inverse.append(vec)
    
    if(len(matrix) == 2):
        x = (matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0])
        if x==0:
            return
        co = 1.0/x
        for i in range(2):
            vec = []
            for j in range(2):
                val = co * matrix[i][j]
                if i!=j:
                    val *= -1
                vec.append(val)    
            inverse.append(vec)
            
        #swap
        inverse[0][0],inverse[1][1] = inverse[1][1], inverse[0][0]

        
        
    return inverse
    
    