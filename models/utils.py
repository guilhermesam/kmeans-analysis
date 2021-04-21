def split_x_y(matrix):
    """
    Separa um array numpy baseado em colunas para treinamento
    de modelos de ML. Para os propósitos deste treinamento, a
    variável dependente tem sempre a posição -1.

    :param matrix: uma matriz numpy
    :return x: variáveis independentes
    :return y: variável dependente
    """
    
    x = matrix[:,:-1]
    y = matrix[:,-1:]

    return x,y


def get_column(matrix,index):
    """
    obtém a coluna de uma matriz passada 
    por parâmetro

    :param array: uma matriz numpy
    :param index: o índice da coluna desejada
    :return column: a coluna desejada
    """

    return matrix[:,index]