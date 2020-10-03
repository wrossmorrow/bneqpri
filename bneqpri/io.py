"""

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

    Copyright W. Ross Morrow (morrowwr@gmail.com), 2020+

"""

import numpy as np

def read_data_files( firms , products , utilities , budget=False ):

    with open( firms , "r" ) as file: 
        headers = file.readline()
        content = [ f.strip() for f in file.readline().split(',') ]
        F = int(content[0])
        Jf = np.zeros(F,dtype=int)
        for f in range(F): 
            Jf[f] = int(content[f+1])
            
    with open( products, "r" ) as file: 
        headers = file.readline()
        content = [ f.strip() for f in file.readline().split(',') ]
        J = int(content[0])
        c = np.zeros(J)
        for j in range(J): 
            c[j] = float(content[1+j])

    I = sum( 1 for line in open( utilities , "r" ) ) - 1
    
    if budget: 

        a, b, V = np.zeros(I), np.zeros(I), np.zeros((I,J))
        with open( utilities , "r" ) as file: 
            headers = file.readline()
            line , i = file.readline() , 0
            while line: 
                content = [ f.strip() for f in line.split(',') ]
                #print( line )
                b[i] = float(content[0])
                a[i] = float(content[1])
                #print( b0[i] , a0[i] )
                for j in range(J): 
                    V[i,j] = float(content[2+j])
                line , i = file.readline() , i+1
        return I, J, F, Jf, b, a, V, c

    else: 

        a, V = np.zeros(I), np.zeros((I,J))
        with open( utilities , "r" ) as file: 
            headers = file.readline()
            line , i = file.readline() , 0
            while line: 
                content = [ f.strip() for f in line.split(',') ]
                a[i] = float(content[0])
                for j in range(J): 
                    V[i,j] = float(content[1+j])
                line , i = file.readline() , i+1
        return I, J, F, Jf, a, V, c

def read_initial_prices( prices ):

    with open( prices , "r" ) as file: 
        headers = file.readline()
        content = [ f.strip() for f in file.readline().split(',') ]
        p = np.zeros(len(content))
        for j in range(len(content)): 
            p[j] = float(content[j])
        return p

def write_prices( filename , p ): 

    J = p.size
    with open( filename , "w" ) as file: 
        file.write(f"price_0")
        for j in range(1,J):
            file.write(f",price_{j}")
        file.write("\n")
        file.write(f"{p[0]}")
        for j in range(1,J):
            file.write(f",{p[j]}")
        file.write("\n")
