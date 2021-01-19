#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 22:14:59 2020

@author: didier
"""

import networkx as nx
import numpy as np
import math
import os, sys, getopt
import multiprocessing as mp
from multiprocessing import Process, Lock, Queue
from multiprocessing import RawArray, Array

from weave import inline
import time

#from memory_profiler import profile   
#fp=open('memory_profilerPD.log','w+')
#import pandas as pd
#from scipy.stats import pearsonr, spearmanr
#import LinkPredRun as lp
#from copy import deepcopy
#sys.stdout.flush()

factors = [0.05, 0.1, 0.2]
maxSteps = 22


def get_progressive_diffusion_row_code():
    code = r"""        
    if(node >= 0){    
        int j = 0;
        long long i = 0;
        int t0 = 0;
        int t1 = 0;
        It[node] = 1.0;
        ItS[int(node)] = 0.0; 
        float dInf = 0.0;
                
        float * neg_p = NULL;        
        neg_p = (float *) malloc(sizeof(float) * (2*N));
        
        if(neg_p != NULL) {            
            for (i = 0 ; i < N ; i++){
                neg_p[i] = 1.0;    
                neg_p[i+N] = 1.0;  
            }
                        
            for(j = 0; j < steps; j++){  
                //printf("j = %d \n",j);
                t0 = t1;
                t1 = (t0+1) % 2;
                
                for(i=0; i < M; i++){
                   neg_p[nodex[i]] = neg_p[nodex[i]] * ( 1.0 - (iDegree[neig[i]] * It[neig[i]+ t0*N]) );
                   neg_p[nodex[i]+N] -= (iDegree[nodex[i]+N] * (ItR[neig[i]+t0*N] + It[neig[i]+t0*N]));                    
                }
                
                for (i = 0 ; i < N ; i++){                    
                    ItS[i+t1*N] = ItS[i+t0*N] * neg_p[i];
                    It[i+t1*N] = (It[i+t0*N] * neg_p[i+N] ) + (ItS[i+t0*N]*(1.0 - neg_p[i] ));
                    ItR[i+t1*N] = ItR[i+t0*N] + (It[i+t0*N]*(1.0 - neg_p[i+N]));
                    
                    neg_p[i] = 1.0;    
                    neg_p[i+N] = 1.0;                      
                }
     
            }
                
            free(neg_p);
            return_val = t1;
        }
        else
            return_val = -9;            
    }    
    else
        return_val = -1;
        
     """
    return code
    

def compile_get_progressive_diffusion_row():
    
    N = 5
    M = 10
    iDegree =  np.zeros([N,1],dtype=np.float32) 
    neig = np.zeros([N,1],dtype=int)
    nodex = np.zeros([N,1],dtype=int)
    steps = 1
    
    ItR = np.zeros([N,1],dtype=np.float32)
    It = np.zeros([N,1],dtype=np.float32)
    ItS = np.zeros([N,1],dtype=np.float32)
    node = -1    
        
    code = get_progressive_diffusion_row_code() 
    t1 = inline(code,['iDegree','ItR','It','ItS', 'steps','N','node','M','nodex','neig'],
                      compiler='gcc')
    
    return True

compile_get_progressive_diffusion_row()



#@profile(stream=fp)
def worker_job_score(lock, task_queue, shared_array, shared_Degree, shared_neig, shared_nodex, steps, probMatrix='I'):
      
    countTentatives = 0
    readFirstTime = False
    proc_name = mp.current_process().name  
    N = int(len(shared_Degree)/2)
    M = len(shared_neig)
    
    X_shape = (N,N)
    #X_np = np.frombuffer(result_list).reshape(X_shape) 
#    with lock:
    X_np = np.frombuffer(shared_array.get_obj(),dtype=np.float32).reshape(X_shape)
    
    Degree = np.frombuffer(shared_Degree,dtype=np.float32).reshape((2*N,1))   
    neig = np.frombuffer(shared_neig,dtype=np.long).reshape((M,1))
    nodex = np.frombuffer(shared_nodex,dtype=np.long).reshape((M,1))
    
    while True:
        try:
            next_task = task_queue.get_nowait()
                       
        except:
            if countTentatives < 3 and not readFirstTime:
                time.sleep(5)   
                print(str(proc_name)+': trying to read the Queue Again!')
                countTentatives = countTentatives +1
                pass
            else:
                print(str(proc_name)+': Exiting- Queue empty!')            
                break
        else:            
            readFirstTime = True
            nodes = next_task 
            print(nodes[0], nodes[-1])
            
            for node in nodes:                
                resp = getNodeAnalyticDiffModelFunctionOPT(node, Degree, neig, nodex, steps)                
                val = resp['I'].flatten()
                
                if probMatrix == 'I': 
                    np.copyto(X_np[node,:],val)
                    
                    
                else:
                    val =  val + resp['R'].flatten()
                    np.copyto(X_np[node,:],val)
                            
                            
    return True



#@jit
def getListGraphB(A):    
    N = A.shape[0]
    B = np.array(A == 1, dtype=bool)
    indices = np.linspace(0,A.shape[0]-1,A.shape[0],dtype=long)
    glist = np.empty((N,1),dtype= np.object)    
    glist[:,0] = [indices[np.take(B,[x],axis=0).flatten()] for x in xrange(N)]
    return glist


def getExtraParameters(glist,beta,gamma):
    N = np.int(len(glist))
    ldeg = np.array([1.0/len(glist.item(x)) for x in xrange(len(glist))],dtype=np.float32)
    Degree = np.concatenate((beta*ldeg,gamma*ldeg), axis=None)
    neig = np.empty((0,0),dtype=long)
    nodex = np.empty((0,0),dtype=long)
    
    for x in xrange(N):
        neig = np.concatenate((neig,glist.item(x)), axis=None)
        nodex = np.concatenate((nodex,np.full((glist.item(x).size, ), x)), axis=None)  
    
    sDegree = np.ctypeslib.as_ctypes(Degree)
    sneig = np.ctypeslib.as_ctypes(neig)
    snodex = np.ctypeslib.as_ctypes(nodex)
    
    del Degree, neig, nodex
    
    return sDegree, sneig, snodex


#@profile(stream=fp)
def get_progressive_diffusion_row(iDegree, neig, nodex, steps,node):
    
    code = get_progressive_diffusion_row_code()
     
    def getNumpyState(ItR,It):
        state = dict()    
        state['I'] = It
        #state['S'] = ItS
        state['R'] = ItR
        return state
    
    
    N = np.int(len(iDegree)/2)
    M = len(nodex)

    ItR = np.zeros([2*N,1],dtype=np.float32) 
    It = np.zeros([2*N,1],dtype=np.float32)        
    ItS = np.ones([2*N,1],dtype=np.float32)
        
    t1 = inline(code,['iDegree','ItR','It','ItS', 'steps','N','node','M','nodex','neig'],
                      compiler='gcc')
    
    ItR = ItR.reshape((2,N)).T
    It = It.reshape((2,N)).T
    
    del ItS
    
    
    iterator2 = getNumpyState(ItR[:,t1],It[:,t1])
    return iterator2


def getNodeAnalyticDiffModelFunctionOPT( 
        node, 
        Degree, 
        neig,nodex,
        steps = maxSteps):    
          
    iterator = get_progressive_diffusion_row(Degree, neig, nodex, steps,node)    
    return iterator



def getPredictedLinks(xcorr, m0):
    edges = list()
    corrLinks = list()   
    
    ind = np.array(np.unravel_index(np.argsort(xcorr, axis=None), xcorr.shape))
    tam = len(ind[0])
   
    
    for i in xrange(tam-1,tam-1-m0,-1):

        ed = (ind[0][i],ind[1][i])
        edges.append(ed)
        corrLinks.append(xcorr[ed])
        
    return edges, corrLinks

# =============================================================================
#     
# 
#     USAR E VER ARQUIVO LP-DiffusionPerformanceTest.py
#     E FAZER OS TESTES EM lpd-PerformanceTest
# 
# 
# =============================================================================


#@profile(stream=fp)
def getProgressiveDiffusionScoreOPT(glist, beta, gamma, steps = maxSteps, probMatrix='I', nworkers=-1):    
    
    N = len(glist)
    compile_get_progressive_diffusion_row()    
    
    
    if nworkers < 0:
        num_workers = int(mp.cpu_count())
    else:
        num_workers = nworkers   
    

    numPJob = int(math.floor(N/float(num_workers)))
    numPJob = 1 if numPJob == 0 else numPJob
         
    
    Degree, neig, nodex = getExtraParameters(glist,beta, gamma)
    shared_Degree = RawArray(Degree._type_, Degree)
    shared_neig = RawArray(neig._type_, neig)
    shared_nodex = RawArray(nodex._type_, nodex)
    
    del Degree, neig, nodex
       
    task_queue = Queue()   
    lock = Lock()  
    X_shape = (N,N)
    shared_array = Array('f', X_shape[0] * X_shape[1],lock=lock)  
    #shared_array = sharedctypes.RawArray(data._type_, data)
    
    # Wrap X as an numpy array so we can easily manipulates its data.
    X_np = np.frombuffer(shared_array.get_obj(),dtype=np.float32).reshape(X_shape)

    # Copy data to our shared array.
    data = np.zeros((N, N), dtype=np.float32)
    np.copyto(X_np, data) 
    del data
    
   
    listW = list()    
    for i in xrange(N):        
        listW.append(i)                
        if len(listW) >= numPJob:
            task_queue.put(listW)            
            listW = list()
            
    if len(listW) > 0:
        task_queue.put(listW)
    
    # Start consumers
    processes = []
    
    print ('Creating '+str(num_workers)+' workers')
    
    for w in xrange(num_workers):        
        p = Process(target=worker_job_score, args=(lock, task_queue, shared_array, shared_Degree, shared_neig, shared_nodex,steps, probMatrix))        
        processes.append(p)
        p.start()
        
    for w in processes:
        w.join()  
    print(" Returning..")
    del shared_Degree, shared_neig, shared_nodex, listW, processes
    #print(np.sum(X_np, 1))    
    
    
    return X_np 


#@profile(stream=fp)
def getProgressiveDiffusion_matrixOPT(
        g, # the nx graph to be processed        
        beta=0.1, #the probability of infection
        gamma=1.0, # the probability of recovering
        steps = 22, #max number of diffusion steps
        nworkers = -1, #-1 to use all available threads in the system or the number of desired workers
        probMatrix='I' # I for the currently infected prob., otherwise the Recovery probability matrix
        ):
    
# =============================================================================
#     #remove isolated nodes
#     g.remove_nodes_from(list(nx.isolates(g)))     
# =============================================================================
   # N = len(g.nodes)
    
    #relabel the ids
# =============================================================================
#     mapping=dict(zip(g.nodes(),range(0,N)))
#     g = nx.relabel_nodes(g,mapping,False)
#     
# =============================================================================
    Y = nx.adjacency_matrix(g)
    Y = Y.astype(dtype=np.int8)
    A = Y.toarray()

   # A = nx.to_numpy_matrix(g,dtype=np.int8)    
    glist = getListGraphB(A)
    
    
    del A, Y
    
    
    listRS = getProgressiveDiffusionScoreOPT(glist, beta, gamma, steps, probMatrix, nworkers)            
    return listRS
     


#DAVO: REfatorar RECEBENDO UMA FUNCAO E FAZENDO COMO EM TESTES LAVI PARALLEL
def worker_job(lock, task_queue, result_list, result_listI, g, beta, gamma, steps):
       
    proc_name = mp.current_process().name
    Degree = np.array([[beta/g.degree[x], gamma/g.degree[x]] for x in  g.nodes], dtype=np.float32)
    while True:
        try:
            next_task = task_queue.get_nowait()
        
        except:
            print(str(proc_name)+': Exiting- Queue empty!') 
            break
        else:            
            nodes = next_task 
            print(nodes[0], nodes[-1])
            listS = dict()
            listI = dict()
            
            for node in nodes:                
                resp = getNodeAnalyticDiffModelFunctionOLD(g, node, Degree, steps)
                listS[node] = list(1 - resp['S'])
                listI[node] = list(resp['I'])
            #resp = dict()
            #resp[node] = list(ivalues)
            with lock:
                result_list.update( listS )                     
                result_listI.update( listI )
                #result_list.extend(resp)
    return True


def initNumpyStates(g, nstates):
    n_nodes = len(g.nodes())      
    state = dict()    
    state['I'] = np.zeros([n_nodes,nstates],dtype=np.float32)        
    state['S'] = np.ones([n_nodes,nstates],dtype=np.float32)
    state['R'] = np.zeros([n_nodes,nstates],dtype=np.float32)
    return state
    
    

#TODO: DAVO: REfatorar usando weave e C. IMPLEMENTAR SEM GUARDAR TODOS OS STEPS E USANDO O NOVO
def getNodeAnalyticDiffModelFunctionOLD(
        G, 
        node, 
        Degree, 
        steps = maxSteps):
    g = G
    nodes = g.nodes()
    
    #nx.diameter(g)

    iterator = initNumpyStates(g, steps)
    #setting the unique node as initial spreader
    iterator['I'][node,0] = 1.0     
    iterator['S'][node,0] = 0.0 
    iDegree = Degree        
 
    nsteps = range(steps-1)
    
    for t in nsteps: 
     
        It = iterator['I'][:,t]
        ItR = (It + iterator['R'][:,t])
        aa = [[(1.0 - iDegree[list(g[x]),0] * It[list(g[x])]).prod(), (1.0 - (iDegree[x,1] * ItR[list(g[x])] ).sum())]  for x in nodes]
        neg_p = np.array(aa, dtype=np.float32)
      
        #tInf = 0.0
        iterator['S'][:,t+1] = iterator['S'][:,t] * neg_p[:,0]
        iterator['I'][:,t+1] = (iterator['I'][:,t] * neg_p[:,1]) + (iterator['S'][:,t]*(1.0 - neg_p[:,0]))
        #iterator['R'][:,t+1] = iterator['R'][:,t] + (iterator['I'][:,t]*(1.0 - neg_p[:,1]))
        iterator['R'][:,t+1] = iterator['R'][:,t] + (iterator['I'][:,t]*(1.0 - neg_p[:,1]))
        
        #tInf = iterator[t1]['I'].sum()
    return iterator




def getProgressiveDiffusionStepsScore(g, beta, gamma, steps = maxSteps):    
    
    #degree = list(dict(g.degree).values())
    nodes = g.nodes      
    print('beta = ',beta)
    
    mgr = mp.Manager()
    tasks = mgr.Queue()   
    results = mgr.dict() 
    resultsI = mgr.dict()
    lock = mgr.Lock()
    
    listW = list()    
    for i in nodes:
        listW.append(i)                
        if len(listW) >= 20:
            tasks.put(listW)            
            #listW.clear()
            listW *= 0
            
    if len(listW) > 0:
        tasks.put(listW)
    
    # Start consumers
    processes = []
    num_workers = mp.cpu_count()
    print ('Creating '+str(num_workers)+' workers')
    
    for w in range(num_workers):
        p = mp.Process(target=worker_job, args=(lock, tasks, results, resultsI, g, beta, gamma, steps))
        processes.append(p)
        p.start()
        
    for w in processes:
        w.join()  
          
    index = list(results.keys())
    index.sort()
    listrS = [results[x] for x in index]
    
    #index = list(resultsI.keys())
    #index.sort()
    listrI = [resultsI[x] for x in index]
    
    return listrS, listrI  


#TODO: DAVO FAZER VERSAO VARIOS STEPS opt
def getProgressiveDiffusionSteps_matrices(
        g, # the nx graph to be processed
        fileMatrix='', # the name of the file where temporaly saved the matrix (optional)
        beta=0.1, #the probability of infection
        gamma=1.0, # the probability of recovering
        steps = 22 #max number of diffusion steps
        ):
    
    saved = fileMatrix != ''
    
    #remove isolated nodes
    g.remove_nodes_from(list(nx.isolates(g)))     
    N = len(g.nodes)
    
    #relabel the ids
    mapping=dict(zip(g.nodes(),range(0,N)))
    nx.relabel_nodes(g,mapping,False)

    ## list(nx.isolates(g))
#    if len(list(isolates)) > 0:
#        for isolate_node in isolates:
#            g.remove_node(isolate_node)
#   
            
    if not os.path.isfile(fileMatrix+'.npz'):        
        listRS, listI = getProgressiveDiffusionStepsScore(g,beta,gamma,steps)        
    
        if (saved):
            np.savez(fileMatrix, listRS=listRS,listI=listI)
            print('saving the temp matrix in ', fileMatrix )  
    else:
        print('reading the temp matrix from ', fileMatrix )  
        data = np.load(fileMatrix+".npz", allow_pickle = True)
        listRS =  list(data['listRS'])
        listI =  list(data['listI'])
    listRS = [np.array(list(listRS[i]), dtype=np.float) for i,_ in enumerate(listRS)]
    listI = [np.array(list(listI[i]), dtype=np.float) for i,_ in enumerate(listI)]
    nsteps = min([x.shape[1] for x in listRS])  
    matrixL = [listRS, listI]  
    
    return matrixL, nsteps


    
    
def processGmlFile(graphFile, graphFolder, beta, gamma):

    print("graphFile " + graphFile + ", beta, gamma ", beta, gamma) 
    temp = os.path.splitext(graphFile)    
    fileName = (temp[0].split(os.sep)[-1], temp[1])
    g = nx.read_gml(graphFile,'id')

    N = len(g.nodes)
    M = len(g.edges)
    m0 = int(M*0.5) 
    A = nx.to_numpy_matrix(g,dtype=np.int8) + np.identity(N,dtype=np.int8)*10
    lambdaP = str(beta/gamma).replace('.', '_')
    
    fileMatrix = os.path.join(graphFolder, fileName[0] +"MATRIXS-Peak_"+lambdaP)
    
#    print(fileMatrix)
#    if not os.path.isfile(fileMatrix):
#        listRS, listI = getProgressiveDiffusionStepsScore(g,beta,gamma)        
#        fileMatrix = os.path.join(graphFolder, fileName[0] +"MATRIXS-Peak_"+lambdaP)
#        np.savez(fileMatrix, listRS=listRS,listI=listI)
#        
#    else:
#        data = np.load(fileMatrix, allow_pickle = True)
#        listRS =  list(data['listRS'])
#        listI =  list(data['listI'])
#    
#    listRS = [np.array(list(listRS[i]), dtype=np.float) for i,_ in enumerate(listRS)]
#    listI = [np.array(list(listI[i]), dtype=np.float) for i,_ in enumerate(listI)]
#    nsteps = min([x.shape[1] for x in listRS])  
#    matrixL = [listRS, listI]   
#    
    matrixL, nsteps = getProgressiveDiffusionSteps_matrices(g, fileMatrix, beta, gamma)
    
    cli = ['Eta','Inf']
    
    for step in xrange(3,nsteps,3):
        print (step) 
        
        for li in xrange(0,2):
        
            listM = [x[:,step] for x in matrixL[li]]                
            RP_DF = np.triu(np.array(listM) + np.array(listM).transpose() - (A*10))
            RP_DF2 = np.triu(np.array(listM)/2 + np.array(listM).transpose()/2 - (A*10))
            
            edges, _ = getPredictedLinks(xcorr=RP_DF, m0 = m0)
            edges2, _ = getPredictedLinks(xcorr=RP_DF2, m0 = m0)
        
        
            for factor in factors:
                newM = int(math.floor(M*(factor)))            
        
                top = edges[0:newM]
                GG = g.copy()            
                GG.add_edges_from(top)
                newFile = os.path.join(graphFolder, fileName[0] +"__"+cli[li]+"Pt"+str(step)+"-RPS" + str(factor).replace('.', '-') + "L"+lambdaP + fileName[1])
                nx.write_gml(GG, newFile) 
                
                top2 = edges2[0:newM]
                GG = g.copy()            
                GG.add_edges_from(top2)
                newFile = os.path.join(graphFolder, fileName[0] +"__"+cli[li]+"Pt"+str(step)+"-RPA" + str(factor).replace('.', '-') + "L"+lambdaP + fileName[1])
                nx.write_gml(GG, newFile) 
                

    print("FINISH2 ")

    return top, top2



if __name__ == "__main__":
   ## g = nx.read_gml('test.gml','id')
   ## listM =  np.array([[0.1,0.2,0.3,0.4],[0.2,0.4,0.6,0.8],[0.1,0.3,0.5,0.7],[1,1,1,1]])  
    argv = sys.argv[1:]
    
    
# =============================================================================
#     import time
#     from memory_profiler import memory_usage
#     start_time = time.time()
#     
#     listRS = getProgressiveDiffusionScoreOPT(glist, beta, gamma, steps, probMatrix, nworkers)            
#     
#     mem_usage = memory_usage((getProgressiveDiffusionScoreOPT, (glist, beta, gamma, steps, probMatrix, nworkers)),
#                                multiprocess=True,include_children=True, max_usage=True )
#     print('Memory usage (in chunks of .1 seconds): %s' % mem_usage)
#     print('Maximum memory usage: %s' % max(mem_usage))
#     
#     runtime = time.time() - start_time
#     print('runtime',runtime)
#     print(np.sum(listRS, 1))  
#     #aaa = func_to_profile(getProgressiveDiffusionScoreOPT,glist, beta, gamma, steps, probMatrix, nworkers)
#     #aaa.dump_stats('aaa.txt')     
#    
# =============================================================================
    
    try:
        opts, args = getopt.getopt(argv,"h:",["help"])
    except getopt.GetoptError:
        print ('program.py <gml path+file > <folder path result > beta gamma')        
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print ('program.py <gml path+file > <folder path result > beta gamma')            
            sys.exit()
    
    fileGml = str(argv[0])
    graphFolder = str(argv[1])
    beta = float(argv[2])
    gamma = float(argv[3])
  #  fileGml = graphFile = '.\\bases\\BA__10000-12.gml'
  #  graphFolder = '.\\bases'
  
    processGmlFile(fileGml, graphFolder, beta, gamma)
    
    
    

