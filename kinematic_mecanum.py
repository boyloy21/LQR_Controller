import numpy as np

class mecanum:
    def __init__ (self,r,lx,ly):
        self.r=r
        self.lx=lx
        self.ly=ly
    def inverse_kinematic(self,vx,vy,w):
        #Motor1
        w1=(vx-vy-(self.lx+self.ly)*w)/self.r
        #Motror2
        w2=(vx+vy+(self.lx+self.ly)*w)/self.r
        #Motor3
        w3=(vx+vy-(self.lx+self.ly)*w)/self.r
        #Motor4
        w4=(vx-vy+(self.lx+self.ly)*w)/self.r
        return w1,w2,w3,w4
    def forward_kinematic(self,w1,w2,w3,w4):
        #Longitudinal_velocity
        Vx=(w1+w2+w3+w4)*self.r/4
        #Transversal Velocity
        Vy=(-w1+w2+w3-w4)*self.r/4
        #Angular Velocity
        Wz=(-w1+w2-w3+w4)*self.r/(4*(self.lx+self.ly))
        return Vx,Vy,Wz
    def Rotation_matrix(self,theta,vx,vy,omega):
        R = np.array([[np.cos(theta),np.sin(theta),0.0],
                     [-np.sin(theta), np.cos(theta), 0.0],
                     [0.0, 0.0, 1.0]])
        V = R @ np.array([[vx],[vy],[omega]])
        return V
    def discrete_state(self,x,y,yaw,w1,w2,w3,w4,dt):
        dx,dy,dyaw=self.forward_kinematic(w1,w2,w3,w4)
        x_next = x + dx * dt
        y_next = y + dy * dt
        yaw_next = yaw + dyaw * dt

        return x_next, y_next, yaw_next
    
if __name__ == "__main__":
    mecanum()