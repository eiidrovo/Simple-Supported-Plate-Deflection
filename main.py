import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from sklearn.linear_model import LinearRegression 


###################################################################
############### PLOT EXPERIMENTAL VALUES###########################
###################################################################

########### PLOT EXPERIMENTAL VALUES
# FIRST EXPERIMENT
file=open('ss1.txt')
time=[]
s1=[] 
s3=[] 
s5=[] 
s7=[] 
file.readline()
for line in file:
    line=line.strip('\n').split('\t')
    time.append(float(line[0]))
    s1.append(float(line[1])*10**-6)
    s3.append(float(line[2])*10**-6)
    s5.append(float(line[3])*10**-6)
    s7.append(float(line[4])*10**-6)
file.close


fig, axes = plt.subplots(1,2)
fig.suptitle('Experiment 1. Strains', fontsize=12)
fig.supxlabel('Time [s]',fontsize=8)
fig.supylabel('Unit deformation',fontsize=8)
axes[0].set_title("Strains in X", fontsize=8)
axes[0].scatter(time, s1,c='lightcoral', label='Strain1')
axes[0].scatter(time, s3,c='aquamarine', label='Strain3')
axes[0].legend(loc='upper left', prop={'size': 6})

axes[1].set_title("Strains in Y", fontsize=8)
axes[1].scatter(time, s5,c='plum', label='Strain5')
axes[1].scatter(time, s7,c='wheat', label='Strain7')
axes[1].legend(loc='lower right', prop={'size': 6})

plt.savefig('figures/experimental/exp1.png')


########### PLOT EXPERIMENTAL VALUES
# FIRST EXPERIMENT
file=open('ss2.txt')
time=[]
s1=[] 
s3=[] 
s5=[] 
s7=[] 
file.readline()
for line in file:
    line=line.strip('\n').split('\t')
    time.append(float(line[0]))
    s1.append(float(line[1])*10**-6)
    s3.append(float(line[2])*10**-6)
    s5.append(float(line[3])*10**-6)
    s7.append(float(line[4])*10**-6)
file.close

fig, axes = plt.subplots(1,2)
fig.suptitle('Experiment 2. Strains', fontsize=12)
fig.supxlabel('Time [s]',fontsize=8)
fig.supylabel('Unit deformation',fontsize=8)

axes[0].set_title("Strains in X", fontsize=8)
axes[0].scatter(time, s1,c='lightcoral', label='Strain1')
axes[0].scatter(time, s3,c='aquamarine', label='Strain3')
axes[0].legend(loc='upper left', prop={'size': 6})
axes[0].ticklabel_format(style="sci", scilimits=(0,0))

axes[1].set_title("Strains in Y", fontsize=8)
axes[1].scatter(time, s5,c='plum', label='Strain5')
axes[1].scatter(time, s7,c='wheat', label='Strain7')
axes[1].legend(loc='lower right', prop={'size': 6})
axes[1].ticklabel_format(style="sci", scilimits=(0,0))
plt.savefig('figures/experimental/exp2.png')
###################################################################
###################################################################
###################################################################

# Theoretical values
file=open('data.json')
data=json.load(file)
file.close

E=data['E']
pois=data['pois']
a=data["a"]
b=data["b"]
t=data["t"]
xf=data["xf"]
yf=data["yf"]
s1p=data["s1p"] #x
s3p=data["s3p"] #x
s5p=data["s5p"] #y
s7p=data["s7p"] #y
w1=data["w1"]*9.8
w2=9.8*data["w2"]+w1
w3=9.8*data["w3"]+w2
w4=9.8*data["w4"]+w3
series=data["series"]
D=E*t*3/(12*(1-pois*2))
pi=np.pi
forces=np.array([0,w1,w2,w3,w4])

# parameters
x=np.arange(0,a,a/50)
y=np.arange(0,b,b/50)
coords=np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
mn=np.arange(1,series)
mn=np.array(np.meshgrid(mn, mn)).T.reshape(-1, 2)


#Functions necessary and plots in 3d
pmn= lambda m,n,f:(4*f/(a*b))*np.sin(m*pi*xf/(a))*np.sin(n*pi*yf/(b))

def plot_load (forces_arr):
    force_index=0
    for force in forces_arr:
        
        load=np.zeros((np.shape(coords)[0],3))
        for c in range(np.shape(coords)[0]):
            load[c][0]=coords[c][0]
            load[c][1]=coords[c][1]
            s=0
            for n,m in mn:
                s+=pmn(m,n,force)*np.sin(m*pi*coords[c][0]/(a))*np.sin(n*pi*coords[c][1]/(b))
            load[c][2]=s
        #Plot the load

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Load Distribution at {}[N]".format(f'{force:.1}'))
        ax.set_xlabel('X[m]', linespacing=3.2)
        ax.set_ylabel('Y[m]', linespacing=3.2)
        ax.set_zlabel('Load[F/A]', linespacing=3.2)
        surf = ax.plot_trisurf(load[:,0], load[:,1], load[:,2],
        cmap=cm.jet, linewidth=0)
        fig.colorbar(surf, orientation = 'horizontal')
        fig.tight_layout()
        plt.savefig('figures/loads/p {}.png'.format(force_index))
        
        force_index+=1

wmn = lambda m,n,f:pmn(m,n,f)/((D*pi**4)*((m/a)**2+(n/b)**2)**2)

def plot_deformation(forces_arr):
    force_index=0
    for force in forces_arr: 
        w=np.zeros((np.shape(coords)[0],3))
        for c in range(np.shape(coords)[0]):
            w[c][0]=coords[c][0]
            w[c][1]=coords[c][1]
            s=0
            s2=0
            nm_counter=0
            for n,m in mn:
                s+=wmn(m,n,force)*np.sin(m*pi*coords[c][0]/(a))*np.sin(n*pi*coords[c][1]/(b))
            nm_counter+=1
            w[c][2]=s
        #Plot the defflection
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Deformation Distribution at {}[N]".format(f'{force:.1}'))
        ax.set_xlabel('X[m]', linespacing=3.2)
        ax.set_ylabel('Y[m]', linespacing=3.2)
        ax.set_zlabel('Z[m]', linespacing=3.2)
        surf = ax.plot_trisurf(w[:,0], w[:,1], w[:,2], cmap=cm.jet,linewidth=0)
        fig.colorbar(surf, orientation = 'horizontal')
        fig.tight_layout()
        plt.savefig('figures/displacement/w {}.png'.format(force_index))
        force_index+=1        

Mx= lambda m,n,x,y,f:(4*f/(pi**2*a*b))*((m/a)**2+pois*(n/b))*(np.sin(m*pi*xf/a)*np.sin(n*pi*yf/b)*np.sin(m*pi*x/a)*np.sin(n*pi*y/b))/((m/a)**2+(n/b)**2)**2

def plot_mx(forces_arr):
    force_index=0
    for force in forces_arr:
        Mx_arr=np.zeros((np.shape(coords)[0],3))
        for c in range(np.shape(coords)[0]):
            Mx_arr[c][0]=coords[c][0]
            Mx_arr[c][1]=coords[c][1]
            s=0
            for n,m in mn:
                s+= Mx(m,n,coords[c][0],coords[c][1],force)
            Mx_arr[c][2]=s

        #Plot Mx
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Moment in X Distribution at {}[N]".format(f'{force:.1}'))
        ax.set_xlabel('X[m]', linespacing=3.2)
        ax.set_ylabel('Y[m]', linespacing=3.2)
        ax.set_zlabel('Mx[F]', linespacing=3.2)
        surf = ax.plot_trisurf(Mx_arr[:,0], Mx_arr[:,1], Mx_arr[:,2], cmap=cm.jet,linewidth=0)
        fig.colorbar(surf, orientation = 'horizontal')
        fig.tight_layout()
        plt.savefig('figures/moment x/mx {}.png'.format(force_index))
        force_index+=1

My= lambda m,n,x,y,f:(4*f/(np.pi**2*a*b))*(data["pois"]*(m/a)**2+(n/b))*(np.sin(m*np.pi*xf/a)*np.sin(n*np.pi*yf/b)*np.sin(m*np.pi*x/a)*np.sin(n*np.pi*y/b))/((m/a)**2+(n/b)**2)**2

def plot_my(forces_arr):
    force_index=0
    for force in forces_arr:
        My_arr=np.zeros((np.shape(coords)[0],3))
        for c in range(np.shape(coords)[0]):
            My_arr[c][0]=coords[c][0]
            My_arr[c][1]=coords[c][1]
            s=0
            for n,m in mn:
                s+= My(m,n,coords[c][0],coords[c][1],force)
            My_arr[c][2]=s

        #Plot My
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Moment in Y Distribution at {}[N]".format(f'{force:.1}'))
        ax.set_xlabel('X[m]', linespacing=3.2)
        ax.set_ylabel('Y[m]', linespacing=3.2)
        ax.set_zlabel('My[F]', linespacing=3.2)
        surf = ax.plot_trisurf(My_arr[:,0], My_arr[:,1], My_arr[:,2], cmap=cm.jet,linewidth=0)
        fig.colorbar(surf, orientation = 'horizontal')
        fig.tight_layout()
        plt.savefig('figures/moment y/my {}.png'.format(force_index))
        force_index+=1


##  LINEAR REGRESION SIMPLE SUPPORTED CASE THEORETICAL

s1=np.zeros(np.shape(forces)[0])
for c in range(np.shape(forces)[0]):
    for m,n in mn:
        s1[c]+=(6*Mx(m,n,s1p[0],s1p[1],forces[c])/t**2)/E-pois*(6*My(m,n,s1p[0],s1p[1],forces[c])/t**2)/E

s3=np.zeros(np.shape(forces)[0])
for c in range(np.shape(forces)[0]):
    for m,n in mn:
        s3[c]+=(6*Mx(m,n,s3p[0],s3p[1],forces[c])/t**2)/E-pois*(6*My(m,n,s3p[0],s3p[1],forces[c])/t**2)/E

s5=np.zeros(np.shape(forces)[0])
for c in range(np.shape(forces)[0]):
    for m,n in mn:
        s5[c]+=pois*(6*Mx(m,n,s5p[0],s5p[1],forces[c])/t**2)/E-(6*My(m,n,s5p[0],s5p[1],forces[c])/t**2)/E
s7=np.zeros(np.shape(forces)[0])
for c in range(np.shape(forces)[0]):
    for m,n in mn:
        s7[c]+=pois*(6*Mx(m,n,s7p[0],s7p[1],forces[c])/t**2)/E-(6*My(m,n,s7p[0],s7p[1],forces[c])/t**2)/E


regresion_lineal = LinearRegression()
regresion_lineal.fit(forces.reshape(-1,1), s1)
Theory_S1= lambda m : regresion_lineal.coef_*m+regresion_lineal.intercept_
TS1=Theory_S1(forces)
s1label='S1={}x+{}'.format(f'{float(regresion_lineal.coef_):.2e}',f'{float(regresion_lineal.intercept_):.2e}')

regresion_lineal = LinearRegression()
regresion_lineal.fit(forces.reshape(-1,1), s3)
Theory_S3= lambda m : regresion_lineal.coef_*m+regresion_lineal.intercept_
TS3=Theory_S3(forces)
s3label='S5={}x+{}'.format(f'{float(regresion_lineal.coef_):.2e}',f'{float(regresion_lineal.intercept_):.2e}')


regresion_lineal = LinearRegression()
regresion_lineal.fit(forces.reshape(-1,1), s5)
Theory_S5= lambda m : regresion_lineal.coef_*m+regresion_lineal.intercept_
TS5=Theory_S5(forces)
s5label='S7={}x+{}'.format(f'{float(regresion_lineal.coef_):.2e}',f'{float(regresion_lineal.intercept_):.2e}')



regresion_lineal = LinearRegression()
regresion_lineal.fit(forces.reshape(-1,1), s7)
Theory_S7= lambda m : regresion_lineal.coef_*m+regresion_lineal.intercept_
TS7=Theory_S7(forces)
s7label='S1={}x+{}'.format(f'{float(regresion_lineal.coef_):.2e}',f'{float(regresion_lineal.intercept_):.2e}')


fig, axes = plt.subplots(1,2)
fig.suptitle('Theoretical Values for Strains', fontsize=10)
fig.supxlabel('Force [N]',fontsize=8)
fig.supylabel('Unit deformation',fontsize=8)
axes[0].set_title("Strains in X", fontsize=8)
axes[0].scatter(forces, s1,c='lightcoral', label='Strain1')
axes[0].plot(forces, TS1,c='lightcoral', label=s1label)
axes[0].scatter(forces, s3,c='aquamarine', label='Strain3')
axes[0].plot(forces, TS3,c='aquamarine', label=s3label)
axes[0].legend(loc='upper left', prop={'size': 6})
axes[0].ticklabel_format(style="sci", scilimits=(0,0))
axes[1].set_title("Strains in Y", fontsize=8)
axes[1].scatter(forces, s5,c='plum', label='Strain5')
axes[1].plot(forces, TS5,c='plum', label=s5label)
axes[1].scatter(forces, s7,c='wheat', label='Strain7')
axes[1].plot(forces, TS7,c='wheat', label=s7label)
axes[1].legend(loc='lower right', prop={'size': 6})
axes[1].ticklabel_format(style="sci", scilimits=(1,0))
plt.savefig('figures/theoretical/theory.png')



##  LINEAR REGRESION SIMPLE SUPPORTED CASE. EXPERIMENT 2
file=open('ss1.txt')
s1=[] 
s3=[] 
s5=[] 
s7=[] 
file.readline()
for line in file:
    line=line.strip('\n').split('\t')
    s1.append(float(line[1])*10**-6)
    s3.append(float(line[2])*10**-6)
    s5.append(float(line[3])*10**-6)
    s7.append(float(line[4])*10**-6)
file.close

s1=np.array(s1[:len(s1)//2])
s1=np.array_split(s1,5)
for i in range(5):
    s1[i]=np.quantile(s1[i],0.5)

regresion_lineal = LinearRegression()
regresion_lineal.fit(forces.reshape(-1,1), s1)
Theory_S1= lambda m : regresion_lineal.coef_*m+regresion_lineal.intercept_
TS1=Theory_S1(forces)
s1label='S1={}x+{}'.format(f'{float(regresion_lineal.coef_):.2e}',f'{float(regresion_lineal.intercept_):.2e}')

s3=np.array(s3[:len(s3)//2])
s3=np.array_split(s3,5)
for i in range(5):
    s3[i]=np.quantile(s3[i],0.5)

regresion_lineal = LinearRegression()
regresion_lineal.fit(forces.reshape(-1,1), s3)
Theory_S3= lambda m : regresion_lineal.coef_*m+regresion_lineal.intercept_
TS3=Theory_S3(forces)
s3label='S3={}x+{}'.format(f'{float(regresion_lineal.coef_):.2e}',f'{float(regresion_lineal.intercept_):.2e}')

s5=np.array(s5[:len(s5)//2])
s5=np.array_split(s5,5)
for i in range(5):
    s5[i]=np.quantile(s5[i],0.5)

regresion_lineal = LinearRegression()
regresion_lineal.fit(forces.reshape(-1,1), s5)
Theory_S5= lambda m : regresion_lineal.coef_*m+regresion_lineal.intercept_
TS5=Theory_S5(forces)
s5label='S5={}x+{}'.format(f'{float(regresion_lineal.coef_):.2e}',f'{float(regresion_lineal.intercept_):.2e}')

s7=np.array(s7[:len(s7)//2]) 
s7=np.array_split(s7,5)
for i in range(5):
    s7[i]=np.quantile(s7[i],0.5)

regresion_lineal = LinearRegression()
regresion_lineal.fit(forces.reshape(-1,1), s7)
Theory_S7= lambda m : regresion_lineal.coef_*m+regresion_lineal.intercept_
TS7=Theory_S7(forces)
s7label='S7={}x+{}'.format(f'{float(regresion_lineal.coef_):.2e}',f'{float(regresion_lineal.intercept_):.2e}')

fig, axes = plt.subplots(1,2)
fig.suptitle('Experiment 1. Strains', fontsize=12)
fig.supxlabel('Force [N]',fontsize=8)
fig.supylabel('Unit deformation',fontsize=8)
axes[0].set_title("Strains in X", fontsize=8)

axes[0].scatter(forces, s1,c='lightcoral', label='Strain1')
axes[0].plot(forces, TS1,c='lightcoral', label=s1label)

axes[0].scatter(forces, s3,c='aquamarine', label='Strain3')
axes[0].plot(forces, TS3,c='aquamarine', label=s3label)

axes[0].legend(loc='upper left', prop={'size': 6})
axes[0].ticklabel_format(style="sci", scilimits=(0,0))
axes[1].set_title("Strains in Y", fontsize=8)

axes[1].scatter(forces, s5,c='plum', label='Strain5')
axes[1].plot(forces, TS5,c='plum', label=s5label)

axes[1].scatter(forces, s7,c='wheat', label='Strain7')
axes[1].plot(forces, TS7,c='wheat', label=s7label)
axes[1].legend(loc='lower right', prop={'size': 6})
axes[1].ticklabel_format(style="sci", scilimits=(1,0))
plt.savefig('figures/theoretical/Experimental1.png')
plt.clf()


## Linear regresion of experimental values. Experiment 2.
file=open('ss2.txt')
s1=[] 
s3=[] 
s5=[] 
s7=[] 
file.readline()
for line in file:
    line=line.strip('\n').split('\t')
    s1.append(float(line[1])*10**-6)
    s3.append(float(line[2])*10**-6)
    s5.append(float(line[3])*10**-6)
    s7.append(float(line[4])*10**-6)
file.close

s1=np.array(s1[:len(s1)//2])
s1=np.array_split(s1,5)
for i in range(5):
    s1[i]=np.quantile(s1[i],0.5)

regresion_lineal = LinearRegression()
regresion_lineal.fit(forces.reshape(-1,1), s1)
Theory_S1= lambda m : regresion_lineal.coef_*m+regresion_lineal.intercept_
TS1=Theory_S1(forces)
s1label='S1={}x+{}'.format(f'{float(regresion_lineal.coef_):.2e}',f'{float(regresion_lineal.intercept_):.2e}')




s3=np.array(s3[:len(s3)//2])
s3=np.array_split(s3,5)
for i in range(5):
    s3[i]=np.quantile(s3[i],0.5)

regresion_lineal = LinearRegression()
regresion_lineal.fit(forces.reshape(-1,1), s3)
Theory_S3= lambda m : regresion_lineal.coef_*m+regresion_lineal.intercept_
TS3=Theory_S3(forces)
s3label='S3={}x+{}'.format(f'{float(regresion_lineal.coef_):.2e}',f'{float(regresion_lineal.intercept_):.2e}')

s5=np.array(s5[:len(s5)//2])
s5=np.array_split(s5,5)
for i in range(5):
    s5[i]=np.quantile(s5[i],0.5)

regresion_lineal = LinearRegression()
regresion_lineal.fit(forces.reshape(-1,1), s5)
Theory_S5= lambda m : regresion_lineal.coef_*m+regresion_lineal.intercept_
TS5=Theory_S5(forces)
s5label='S5={}x+{}'.format(f'{float(regresion_lineal.coef_):.2e}',f'{float(regresion_lineal.intercept_):.2e}')

s7=np.array(s7[:len(s7)//2]) 
s7=np.array_split(s7,5)
for i in range(5):
    s7[i]=np.quantile(s7[i],0.5)

regresion_lineal = LinearRegression()
regresion_lineal.fit(forces.reshape(-1,1), s7)
Theory_S7= lambda m : regresion_lineal.coef_*m+regresion_lineal.intercept_
TS7=Theory_S7(forces)
s7label='S7={}x+{}'.format(f'{float(regresion_lineal.coef_):.2e}',f'{float(regresion_lineal.intercept_):.2e}')

fig, axes = plt.subplots(1,2)
fig.suptitle('Experiment 2. Strains', fontsize=12)
fig.supxlabel('Force [N]',fontsize=8)
fig.supylabel('Unit deformation',fontsize=8)
axes[0].set_title("Strains in X", fontsize=8)
axes[0].scatter(forces, s1,c='lightcoral', label='Strain1')
axes[0].plot(forces, TS1,c='lightcoral', label=s1label)
axes[0].scatter(forces, s3,c='aquamarine', label='Strain3')
axes[0].plot(forces, TS3,c='aquamarine', label=s3label)
axes[0].legend(loc='upper left', prop={'size': 6})
axes[0].ticklabel_format(style="sci", scilimits=(0,0))
axes[1].set_title("Strains in Y", fontsize=8)
axes[1].scatter(forces, s5,c='plum', label='Strain5')
axes[1].plot(forces, TS5,c='plum', label=s5label)
axes[1].scatter(forces, s7,c='wheat', label='Strain7')
axes[1].plot(forces, TS7,c='wheat', label=s7label)
axes[1].legend(loc='lower right', prop={'size': 6})
axes[1].ticklabel_format(style="sci", scilimits=(1,0))
plt.savefig('figures/theoretical/Experimental2.png')

#plot_load(forces)
#plt.clf()
#plot_deformation(forces)
#plt.clf()
#plot_mx(forces)
#plt.clf()
plot_my(forces)

