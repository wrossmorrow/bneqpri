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

from bneqpri.solver import FPISolver

class BUFPISolver(FPISolver): 

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
        T = self.b.reshape(self.I,1) - p.reshape(1,self.J)
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

class LUFPISolver(FPISolver):

    def __init__(self, I, J, F, Jf, a, V, c): 
        super().__init__(I, J, F, Jf, None, a, V, c)
        
    def utilities(self, p):
        self.U   = self.a.reshape(self.I,1) * p.reshape(1,self.J) + self.V
        self.DpU = self.a.reshape(self.I,1) * np.ones((1,self.J))
        # self.DppU = np.zeros((I,J)) default, also not needed
        
    def solve(self, **kwargs):
        kwargs['corrected'] = False # don't need to correct linear utilities
        return super().solve(**kwargs)
        