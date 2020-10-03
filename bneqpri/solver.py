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
from time import time

class FPISolver(): 
    
    def __init__(self, I, J, F, Jf, Bi, a, V, c): 
        
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
        
        # internal data
        
        # max income and max income index
        if Bi is not None: 
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
        s  = "The method `utilities` is not implemented in the base class. "
        s += "Please use or define a subclass that implements this method."
        raise Exception(s)
    
    def probabilities(self):
        self.PL , self.P = np.zeros((self.I,self.J)) , np.zeros(self.J)
        uimax = np.maximum( 0 , np.max( self.U , axis=1 ) )
        for i in range(self.I): 
            self.PL[i,:]  = np.exp( self.U[i,:] - uimax[i] )
            self.PL[i,:] /= np.exp( - uimax[i] ) + self.PL[i,:].sum()
        self.P = np.sum( self.PL , axis=0 ) / self.I
        
    def lamgam(self):
        DpUPL = self.DpU * self.PL
        self.L = np.sum( DpUPL , axis=0 ) / self.I
        self.G = self.PL.T.dot( DpUPL ) / self.I

    def zeta_u(self, p, verbose=False):
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
        
    def zeta_c(self, p, verbose=False):
        
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

    def solve(self, p0=None, f_tol=1.0e-6, max_iter=1000, corrected=True, verbose=False, check=False):
        
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
            
            if corrected: 
                self.zeta_c( p , verbose=False )
            else: 
                self.zeta_u( p , verbose=False )
            
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
            
        self.nrms = self.nrms[:self.iter+1]

        return p
