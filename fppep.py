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

import sys
from time import time
import numpy as np

from datetime import datetime as dt

def now():
    return dt.now().isoformat()

def log(msg):
    print( f"{now()} :: {msg}" )

class FPISolver(): 
    
    def __init__(self, I, J, F, Jf, Bi, a, V, c, f_tol=1.0e-6, r_tol=1.0e-6, max_iter=1000): 
        
        """
        
        I:  number of individuals (integer)
        J:  number of products (integer)
        F:  number of firms (integer)
        Jf: number of products per firm (F vector)
        Bi: "budgets" for each individual (I vector)
        a:  price sensitivity for each individual (I vector)
        V:  nonprice utility for each individual/product (I x J matrix)
        c:  costs for each product (J vector)
        
        
        f_tol: the convergence tolerance
        max_iter: the maximum number of iterations
        
        """
        
        # problem data
        self.I  = I  # number of individuals
        self.J  = J  # number of products
        self.F  = F  # number of firms
        self.Jf = Jf # list of the number of products per firm
        self.b  = Bi # individual incomes
        self.a  = a  # price sensitivity
        self.V  = V  # "fized" portion of utility (non-price part)
        self.c  = c  # (unit) costs for each product
        
        # iteration options
        self.f_tol    = f_tol
        self.r_tol    = r_tol
        self.max_iter = max_iter
        
        # internal data
        
        # max income and max income index
        self.maxbi = np.argmax(Bi)
        self.maxb  = Bi[self.maxbi]
        
        # indices segmenting firm blocks
        self.Fis = [ None for f in range(F) ]
        self.Fis[0] = range(0,self.Jf[0])
        for f in range(1,F):
            self.Fis[f] = range(self.Fis[f-1][-1]+1,self.Fis[f-1][-1]+1+self.Jf[f])
        
        # other storage
        self.m    = np.zeros(J)      # markups (prices minus costs)
        self.pr   = np.zeros(self.F) # profits (convenience only)
        self.U    = np.zeros((I,J))  # utilities (for each individual/product)
        self.DpU  = np.zeros((I,J))  # price derivatives of utilities (for each)
        self.DppU = np.zeros((I,J))  # second price derivatives of utilities (fe)
        self.PL   = np.zeros((I,J))  # logit probabilities (idiosyncratic)
        self.P    = np.zeros(J)      # mixed probabilities
        self.L    = np.zeros(J)      # "Lambda" operator, diagonal matrix
        self.G    = np.zeros((J,J))  # "Gamma" operator, dense matrix
        self.z    = np.zeros(J)      # "zeta" map
        self.phi  = np.zeros(J)      # "phi" map, what we want to zero
        
    def utilities(self, p):
        """
        
        Define self.U[i,j], self.DpU[i,j], and self.DppU[i,j]
        for all individuals i and products j as a function of 
        prices p[j], accounting for finite purchasing power 
        in the sense that if p[j] >= b[i], for "budget" b[i], 
        
               U[i,j] ~ -Inf (-1e20 here)
             DpU[i,j] = -Inf (set to zero here)
            DppU[i,j] = lim_{ p -> b[i] } [ DppU[i,j](p) / DpU[i,j](p)^2 ]
            
        For example, if u = a[i] log(b[i]-p) + w[i,j], 
        
               U[i,j] =   a[i] log(b[i]-p) + w[i,j]
             DpU[i,j] = - a[i] / (b[i]-p)
            DppU[i,j] = - a[i] / (b[i]-p)^2
            
        when p < b[i] but 
        
               U[i,j] = - Inf    ** but store -1.0e20 **
             DpU[i,j] = - Inf    ** but store 0 **
            DppU[i,j] = - 1/a[i]
            
        when p >= b[i]. 
        
        This weird storage format is just because we need that 
        limiting ratio of first/second derivatives for the right 
        extension of the fixed point map. See the paper. 
        
        The setting of DpU[i,j] to zero is basically an expression
        of the condition DpU[i,j] PL[i,j] -> 0 as p[j] -> b[i]. 
        This is discussed in Assumption 1 of the paper, but 
        basically ensures sufficient continuity for use of deriv-
        atives for analyzing equilibrium. By setting DpU[i,j] to 
        zero, we avoid numerical issues from other values. 
        
        Technically, we should probably build some confidence that
        values like DpU[i,j] PL[i,j] are _numerically_ continuous 
        as well. 
        
        """
        
        T = self.b.reshape(I,1) - p.reshape(1,J)
        for i in range(self.I): 
            for j in range(self.J): 
                if T[i,j] > 0.0: # tolerance, not just > 0.0? 
                    self.U[i,j]    =   self.a[i] * np.log( T[i,j] ) + self.V[i,j]
                    self.DpU[i,j]  = - self.a[i] / T[i,j]
                    self.DppU[i,j] =   self.DpU[i,j] / T[i,j]
                else:
                    self.U[i,j]    = - 1.0e20
                    self.DpU[i,j]  =   0.0
                    self.DppU[i,j] = - 1.0/self.a[i]
    
    def probabilities(self):
        self.PL , self.P = np.zeros((I,J)) , np.zeros(J)
        uimax = np.maximum( 0 , np.max( self.U , axis=1 ) )
        for i in range(self.I): 
            self.PL[i,:]  = np.exp( self.U[i,:] - uimax[i] )
            self.PL[i,:] /= np.exp( - uimax[i] ) + self.PL[i,:].sum()
        self.P = np.sum( self.PL , axis=0 ) / self.I
        
    def lamgam(self):
        DpUPL = self.DpU * self.PL
        self.L = np.sum( DpUPL , axis=0 ) / self.I
        self.G = self.PL.T.dot( DpUPL ) / self.I
        
    def zeta(self, p, verbose=False):
        
        # z <- inv(diag(LAMp)) * ( \tilde{GAMp}' * m - P )
        # for all prices < maxinc, corrected otherwise
        
        self.utilities(p)
        self.probabilities()
        self.lamgam()
        
        # compute markups (not used elsewhere)
        self.m = p - self.c
        
        # z <- \tilde{GAMp}' * m - P
        for f in range(self.F): 
            fi = self.Fis[f]
            # self.z[fi] = self.G[fi,fi].T.dot( self.m[fi] ) - self.P[fi]
            self.z[fi] = - self.P[fi]
            for j in fi: 
                for k in fi: 
                    self.z[j] += self.G[k,j] * self.m[k]
            
        # nominally z <- inv(diag(LAMp)) * z, but with corrections 
        # for products whose prices are above the population limit
        # on incomes. The correction is
        # 
        #     z[j] = omega[maxinci,j] * ( p[j] - maxinc ) + PL[maxinci,{f}]' * m[{f}]
        # 
        # for j : p[j] > maxb
        for f in range(self.F):
            fi = self.Fis[f]
            prFmi = self.PL[self.maxbi,fi].T.dot( self.m[fi] )
            for j in fi:
                if p[j] > self.maxb: # correction term - price j is too high
                    print( f"WARNING ({self.iter}) -- {p[j]} > {self.maxb} correcting zeta for product {j}" )
                    self.z[j] = self.DppU[self.maxbi,j] * ( p[j] - self.maxb ) + prFmi
                else:
                    if self.L[j] < 0.0: 
                        # some tolerance would be better than "0", like 
                        # LAM[j] < -1.0e-10 (just as an example)
                        self.z[j] /= self.L[j]
                    else: 
                        print( f"WARNING ({self.iter}) -- p[{j}] = {p[j]} < {self.maxb} = maxinc but LAMp[{j}] = {self.L[j]}.")
                        # use a modification of extended map instead of what is calculated above
                        # z[j] = PL[maxinci,{f}]' * m[{f}]
                        self.z[j] = prFmi
                        # we exclude the "DppU[I*j+maxinci] * ( p[j] - maxinc )" term expecting
                        # p[j] to be at least close to maxinc
        
        # compute phi = p - c - z also:
        self.phi = p - self.c - self.z
        
    def profits(self, p, compute_m=False):
        self.utilities(p)
        self.probabilities()
        self.m = p - self.c
        for f in range(self.F):
            fi = self.Fis[f]
            self.pr[f] = self.P[fi].dot( self.m[fi] )
        return self.pr
    
    def combgrad(self, p): 
        self.utilities(p)
        self.probabilities()
        self.lamgam()
        self.m = p - self.c
        cg = self.L * self.m + self.P
        for f in range(self.F):
            fi = self.Fis[f]
            for j in fi: 
                for k in fi: 
                    cg[j] -= self.G[k,j] * self.m[k]
            #G = np.array( self.G[fi,fi].T )
            #cg[fi] -= (self.G[fi,fi].T).dot( self.m[fi] )
        return cg
        
    def combgradz(self, p):
        self.utilities(p)
        self.probabilities()
        self.lamgam()
        self.zeta(p)
        return self.L * self.phi
    
    def probcheck(self, p): 
        
        print( "probcheck: " )
        
        self.utilities(p)
        self.probabilities()
        P = np.array( self.P ) # force a copy, not reference
        
        self.lamgam()
        DP = - np.array( self.G ) # force copy
        for j in range(self.J): 
            DP[j,j] += self.L[j]
        
        df , dh = np.zeros((self.J,self.J)) , np.zeros(10)
        for h in range(10): 
            H = 10**(-h)
            for j in range(self.J): 
                p[j] += H
                self.utilities(p)
                self.probabilities()
                df[:,j] = ( self.P - P ) / H
                p[j] -= H
            dh[h] = np.abs( df - DP ).max()
            print( "  %0.8e: %0.2f %0.10f" % ( H , np.log10(dh[h]) , dh[h] ) )
    
    def gradcheck(self, p): 
        
        print( "gradcheck: " )
            
        pr = np.array( self.profits(p) ) # force a copy, not reference
        cg = self.combgrad(p)
        print( f"  cg - cz: { np.abs( cg - self.combgradz(p) ).max() }" )
        
        df , dh = np.zeros(self.J) , np.zeros(10)
        for h in range(10): 
            H = 10**(-h)
            for f in range(self.F): 
                fi = self.Fis[f]
                for j in self.Fis[f]:
                    p[j] += H
                    self.utilities(p)
                    self.probabilities()
                    prp   = self.P[fi].dot( p[fi] - self.c[fi] )
                    df[j] = ( prp - pr[f] ) / H
                    p[j] -= H
            dh[h] = np.abs( df - cg ).max()
            print( "  %0.8e: %0.2f %0.10f" % ( H , np.log10(dh[h]) , dh[h] ) )

    def solve(self, p0=None, f_tol=None, r_tol=None, max_iter=None, verbose=False, check=False):

        f_tol = self.f_tol if f_tol is None else f_tol
        r_tol = self.r_tol if r_tol is None else r_tol
        max_iter = self.max_iter if max_iter is None else max_iter 
        
        if p0 is None: # random prices in [ c/2 , 3/2c ]
            p = self.c/2.0 + 2.0*np.random.random(J)
        else: 
            p = p0
        
        self.nrms = np.zeros(max_iter)
        self.solved = False
        start = time()
        for self.iter in range(max_iter): 
            
            if check: 
                self.probcheck(p)
                self.gradcheck(p)
            
            self.zeta( p , verbose=False)
            
            # test convergence (using step, not combined gradient)
            self.nrms[self.iter] = np.max( np.abs( self.phi ) )
            
            if verbose: 
                self.profits(p)
                print( f"""
Iteration {self.iter}: 
  min/max price..... {p.min()}, {p.max()}
  min/max profits... {self.pr.min()}, {self.pr.max()}
  marketshare....... {self.P.sum()}
  phi norm.......... {self.nrms[self.iter]}
""" )
                # print( "  prices: " , p )
                
            if self.nrms[self.iter] <= f_tol: 
                self.solved = True
                break
            
            # fixed-point step, equivalently p -> p - phi = p - ( p - c - z )
            p = self.c + self.z

        self.time = time() - start
            
        self.nrms = self.nrms[:self.iter]

        return p
        
class NaiveFPISolver(FPISolver):
        
    def utilities(self, p):
        T = self.b.reshape(I,1) - p.reshape(1,J)
        for i in range(self.I): 
            for j in range(self.J): 
                if T[i,j] > 0.0: 
                    self.U[i,j]    =   self.a[i] * np.log( T[i,j] )#  + self.V[i,j]
                    self.DpU[i,j]  = - self.a[i] / T[i,j] # Dp = a DpT / T*T = - a / T*T
                    self.DppU[i,j] =   self.DpU[i,j] / T[i,j] # self.DpU[i,j] / T
                else: 
                    self.U[i,j]    = - 1.0e20
                    self.DpU[i,j]  =   0.0
                    self.DppU[i,j] = - 1.0 / self.a[i]
        
    def zeta(self, p, verbose=False):
        self.utilities(p)
        self.probabilities()
        self.lamgam()
        self.m = p - self.c
        for f in range(self.F):
            fi = self.Fis[f]
            self.z[fi] = - self.P[fi]
            for j in fi: 
                # self.z[j] = - self.P[j]
                for k in fi: 
                    self.z[j] += self.G[k,j] * self.m[k]
            # self.z[fi] = self.G[fi,fi].T.dot( self.m[fi] ) - self.P[fi]
        self.z = self.z / self.L
        self.phi = p - self.c - self.z
        

def read_data_files( firms , products , utilities ):

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


if __name__ == "__main__": 

    I, J, F, Jf, b, a, V, c = read_data_files(sys.argv[1], sys.argv[2], sys.argv[3])


    print( """
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

Bertrand-Nash Equilibrium Prices with Budgets (Finite Purchasing Power)

Method from: 

  Morrow, W.R. Finite purchasing power and computations of Bertrand–Nash equilibrium prices. 
  Comput Optim Appl 62, 477–515 (2015). https://doi.org/10.1007/s10589-015-9743-7

Note this software is provided AS-IS under the GPL v2.0 License. Contact the author
with any questions. 

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
""" )


    log( f"Preparing data" )
    S = FPISolver(I, J, F, Jf, b, a, V, c)

    log( f"Starting solve" )
    p = S.solve( p0=c , verbose=False )

    if S.solved : 

        log( f"Solved in {S.iter}/{S.max_iter} steps, {S.time} seconds, |p-c-z| = {S.nrms[-1]}" )
        with open( sys.argv[4] , "w" ) as file: 
            file.write(f"{p[0]}")
            for j in range(1,J):
                file.write(f",{p[j]}")
            file.write("\n")
        log( f"(probable) equilibium prices written to {sys.argv[4]}" )

    else: 

        log( f"Failed to solve in {S.max_iter} steps, {S.time} seconds." )

    print( """
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
""" )


