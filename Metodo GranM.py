# encoding=utf-8
import numpy as np  # Python matrix manipulación lib


class Simplex():

    def __init__(self):
                 self._A = ""  # matriz de coeficientes
                 self._b = ""  # Array
                 self._c = ''  # restricción
                 self._B = ''  # Conjunto de subíndices de la variable base
                 self.row = 0  # número de restricciones

    def solve(self):
                 # Lea el contenido del archivo, las dos primeras líneas de la estructura del archivo son el número de variables y el número de restricciones
                 # Lo siguiente es la matriz de coeficientes
                 # Entonces la matriz b
                 # Luego hay restricciones c
                 # Suponga que la forma de programación lineal es la forma estándar (todas las ecuaciones)

        A = []
        b = []
        c = []

        self._A = np.array(A, dtype=float)
        self._b = np.array(b, dtype=float)
        self._c = np.array(c, dtype=float)
        self._A = np.array([[0, -2, 1], [0, 1, -1]], dtype=float)
        self._b = np.array([-2, 1], dtype=float)
        # Coeficiente de restricción de igualdad self._A, vector de columna dimensional 3x1
        self._A = np.array([[1, -1, 1]])
        # Coeficiente de restricción de ecuación self._b, valor 1x1
        self._b = np.array([2])
        #Funcion objetivo
        self._c = np.array([2, 1, 1], dtype=float)
        self._B = []
        self.row = len(self._b)
        self.var = len(self._c)
        (x, obj) = self.Simplex(self._A, self._b, self._c)
        self.pprint(x, obj, A)

    def pprint(self, x, obj, A):
        px = ['x_%d = %f' % (i + 1, x[i]) for i in range(len(x))]
        print(','.join(px))

        print('Función de objetivo mínimo:% f' % obj)
        for i in range(len(A)):
            print('%d-th line constraint value is : %f' % (i + 1, x.dot(A[i])))

    def InitializeSimplex(self, A, b):

                 # obtiene el bi mínimo
        b_min, min_pos = (np.min(b), np.argmin(b))

                 # Convierte todo bi en números positivos
        if (b_min < 0):
            for i in range(self.row):
                if i != min_pos:
                    A[i] = A[i] - A[min_pos]
                    b[i] = b[i] - b[min_pos]
            A[min_pos] = A[min_pos] * -1
            b[min_pos] = b[min_pos] * -1

                 # Agregar variable de holgura
        slacks = np.eye(self.row)
        A = np.concatenate((A, slacks), axis=1)
        c = np.concatenate((np.zeros(self.var), np.ones(self.row)), axis=0)
                 # Todas las variables de holgura se agregan a la base y la solución inicial es b
        new_B = [i + self.var for i in range(self.row)]

                 # El valor de la función objetivo de la ecuación auxiliar
        obj = np.sum(b)

        c = c[new_B].reshape(1, -1).dot(A) - c
        c = c[0]
        # entering basis
        e = np.argmax(c)

        while c[e] > 0:
            theta = []
            for i in range(len(b)):
                if A[i][e] > 0:
                    theta.append(b[i] / A[i][e])
                else:
                    theta.append(float("inf"))

            l = np.argmin(np.array(theta))

            if theta[l] == float('inf'):
                print('unbounded')
                return False

            (new_B, A, b, c, obj) = self._PIVOT(new_B, A, b, c, obj, l, e)

            e = np.argmax(c)

                 # Si la variable artificial todavía está en la base en este momento, reemplácela con la variable original
        for mb in new_B:
            if mb >= self.var:
                row = mb - self.var
                i = 0
                while A[row][i] == 0 and i < self.var:
                    i += 1
                (new_B, A, b, c, obj) = self._PIVOT(new_B, A, b, c, obj, new_B.index(mb), i)

        return (new_B, A[:, 0:self.var], b)

         # Entrada de algoritmo
    def Simplex(self, A, b, c):
        B = ''
        (B, A, b) = self.InitializeSimplex(A, b)

                 # Valor objetivo de la función
        obj = np.dot(c[B], b)

        c = np.dot(c[B].reshape(1, -1), A) - c
        c = c[0]

        # entering basis
        e = np.argmax(c)
                 # Encuentre el número de prueba más grande, si es mayor que 0, la función objetivo se puede optimizar
        while c[e] > 0:
            theta = []
            for i in range(len(b)):
                if A[i][e] > 0:
                    theta.append(b[i] / A[i][e])
                else:
                    theta.append(float("inf"))

            l = np.argmin(np.array(theta))

            if theta[l] == float('inf'):
                print("unbounded")
                return False

            (B, A, b, c, obj) = self._PIVOT(B, A, b, c, obj, l, e)

            e = np.argmax(c)

        x = self._CalculateX(B, A, b, c)
        return (x, obj)

         # Obtenga una solución completa
    def _CalculateX(self, B, A, b, c):

        x = np.zeros(self.var, dtype=float)
        x[B] = b
        return x

         # Transformación base
    def _PIVOT(self, B, A, b, c, z, l, e):
        # main element is a_le
        # l represents leaving basis
        # e represents entering basis

        main_elem = A[l][e]
        # scaling the l-th line
        A[l] = A[l] / main_elem
        b[l] = b[l] / main_elem

        # change e-th column to unit array
        for i in range(self.row):
            if i != l:
                b[i] = b[i] - A[i][e] * b[l]
                A[i] = A[i] - A[i][e] * A[l]

        # update objective value
        z -= b[l] * c[e]

        c = c - c[e] * A[l]

        # change the basis
        B[l] = e

        return (B, A, b, c, z)


s = Simplex()
s.solve()
