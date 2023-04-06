class QLearningMazeFollower:

 

#Backend Methods________________________________________________________________

  def __init__(self,xColumns,yColumns,aislesDictionary):

    import numpy as np
    import plotly.graph_objects as go

    self.np = np
    self.go = go

    self.aislesReward = - 1 #aislesReward: Reward for taking next action to an aisle
    self.destinationReward = 100 #destinationReward: aislesReward: Reward for taking next action to an destination
    self.wallsReward = - 100 #wallsReward: aislesReward: Reward for taking next action to a wall

    self.xColumns = xColumns #xColumns: Number of columns in x starting from 1
    self.yColumns = yColumns #yColumns: Number of columns in y starting from 1
    self.aislesDictionary = aislesDictionary #aislesDictionary: Dictionary consists of aisle values along y axis(Starting from 1) per each key of x axis(Starting from 1)

    self.initializeRewardArray()


  def initializeQLearning(self,discountFactor,greedyPolicy,learningRate):

    self.discountFactor = discountFactor
    self.greedyPolicy = greedyPolicy
    self.learningRate = learningRate

    self.QArray = self.np.zeros((self.xColumns,self.yColumns,4)) #QArray: Array consists of 4 Q values per each cell in maze(Q Value: A measure of the overall expected reward assuming that the Agent continues playing until the end of the episode following some policy)


  def initializeRewardArray(self):

    self.rewardArray = self.np.full((self.xColumns,self.yColumns),self.wallsReward) #rewardArray: Array consists of a reward value per each cell in maze

    for ix in self.aislesDictionary.keys():
      for iy in self.aislesDictionary[ix]:
        self.rewardArray[ix - 1,iy - 1] = self.aislesReward


  def initializeDestination(self,xDestination,yDestination):

    self.currentDestination = (xDestination - 1,yDestination - 1) #currentDestination: Current coordinates of destination as a 1x2 tuple, xDestination: x coordinates of destination, yDestination: y coordinates of destination
    self.rewardArray[self.currentDestination] = self.destinationReward


  def terminalState(self,xCurrent,yCurrent):

    if self.rewardArray[xCurrent,yCurrent] == self.aislesReward:
      return False
    else:
      return True


  def startingLocation(self):

    xStart = self.np.random.randint(self.xColumns) #xStart: Random start point of x including 0
    yStart = self.np.random.randint(self.yColumns) #yStart: Random start point of y including 0
    
    while self.terminalState(xStart,yStart):
      xStart = self.np.random.randint(self.xColumns)
      yStart = self.np.random.randint(self.yColumns)

    return xStart,yStart

    
  def nextAction(self,xCurrent,yCurrent,greedyPolicyCurrent):

    if self.np.random.random() < greedyPolicyCurrent: #greedyPolicy: Probability value between [0,1] to perform the action which should yield the highest expected reward
      return self.np.argmax(self.QArray[xCurrent,yCurrent])
    else:
      return self.np.random.randint(4)

      
  def nextLocation(self,xCurrent,yCurrent,actionIndex):

    if actionIndex == 0 and xCurrent < self.xColumns - 1: #actionIndex = 0: Next action - Right
      xCurrent += 1
    elif actionIndex == 1 and yCurrent < self.yColumns - 1: #actionIndex = 1: Next action - Up
      yCurrent += 1
    elif actionIndex == 2 and xCurrent > 0: #actionIndex = 2: Next action - Left
      xCurrent -= 1
    elif actionIndex == 3 and yCurrent > 0: #actionIndex = 3: Next action - Down
      yCurrent -= 1

    return xCurrent,yCurrent


  def updateQArray(self,xCurrent,yCurrent,xNext,yNext,actionIndex):

    reward = self.rewardArray[xNext,yNext] #reward:  Numerical value received by the Agent from the Environment as a direct response to the Agent’s actions
    QValueCurrent = self.QArray[xCurrent,yCurrent,actionIndex] #QValueCurrent: Q value of a given action index regarding to the current cell
    temporalDifference = reward + (self.discountFactor*self.np.max(self.QArray[xNext,yNext])) - QValueCurrent #temporalDifference: A learning method which learns “on the fly” similar to Monte Carlo, yet updates its estimates like Dynamic Programming, discountFactor: A factor between [0,1] which controls the significance of future rewards with immediate ones
    QValueUpdate = QValueCurrent + (self.learningRate*temporalDifference) #QValueUpdate: Updated Q value of a given action index regarding to the current cell

    self.QArray[xCurrent,yCurrent,actionIndex] = QValueUpdate


  def addObstacle(self,xObstacle,yObstacle):

    self.rewardArray[xObstacle,yObstacle] = self.wallsReward #xObstacle: x coordinate of obstacle, yObstacle: y coordinate of obstacle


  def modelTraining(self,trainingEpisodes):

    for episode in range(trainingEpisodes): #trainingEpisodes: Number of training repetitions
      xCurrent,yCurrent = self.startingLocation()
    
      while not self.terminalState(xCurrent,yCurrent):
        
        actionIndex = self.nextAction(xCurrent,yCurrent,self.greedyPolicy) #actionIndex: Action taken next(0: Right,1: Up,2: Left,3: Down)
        xNext,yNext = self.nextLocation(xCurrent,yCurrent,actionIndex) #xNext: Next coordinate of x, yNext: Next coordinate of y
        
        self.updateQArray(xCurrent,yCurrent,xNext,yNext,actionIndex)

        xCurrent,yCurrent = xNext,yNext
    
    
  def coordinatePathQValuesConstant(self,xStart,yStart):

    xStart -= 1 #xStart: Redefine required starting column of x (Including 0)
    yStart -= 1 #yStart: Redefine required starting column of y (Including 0)

    if self.terminalState(xStart,yStart):
      if self.rewardArray[xStart,yStart] == self.destinationReward:
        return ['Initial Point Identical to the Destination Point']
      else:
        return ['Initial Point Coinciding with the Maze Walls']
    else:
      xPrevious = xStart #xPrevious: Previous x point(Including 0)
      yPrevious = yStart #yPrevious: Previous y point(Including 0)

      actionIndex = self.nextAction(xPrevious,yPrevious,1)
      xCurrent,yCurrent = self.nextLocation(xPrevious,yPrevious,actionIndex) #xCurrent: Current x point(Including 0), yCurrent: Current y point(Including 0)

      coordinatePathArray = [[xPrevious,yPrevious],[xCurrent,yCurrent]]

      while not self.terminalState(xCurrent,yCurrent):
        actionIndex = self.nextAction(xCurrent,yCurrent,1)
        xNext,yNext = self.nextLocation(xCurrent,yCurrent,actionIndex)

        if [xPrevious,yPrevious] == [xNext,yNext]:
          return ['Recursion Detected: Insufficient Training Episodes',coordinatePathArray]

        xPrevious,yPrevious = xCurrent,yCurrent
        xCurrent,yCurrent = xNext,yNext

        coordinatePathArray.append([xCurrent,yCurrent])

      if self.rewardArray[xCurrent,yCurrent] == self.destinationReward:
        return ['Destination Reached',coordinatePathArray]
      else:
        return ['Terminal Detected: Insufficient Training Episodes',coordinatePathArray]
    
    
  def coordinatePathQValuesUpdating(self,xStart,yStart,obstacleArray,intermediateTrainingEpisodes):
 
    xStart -= 1
    yStart -= 1

    if self.terminalState(xStart,yStart):
      if self.rewardArray[xStart,yStart] == self.destinationReward:
        return ['Initial Point Identical to the Destination Point']
      else:
        return ['Initial Point Coinciding with the Maze Walls']
    elif [xStart,yStart] in obstacleArray: #obstacleArray: Array consists of coordinates of obstacle cells in maze(Excluding 0)
      return ['Initial Point Coinciding with a Obstacle Point']
    else:
      currentQArray = self.np.copy(self.QArray)

      xCurrent = xStart
      yCurrent = yStart

      coordinatePathArray = [[xCurrent,yCurrent]]

      while self.rewardArray[xCurrent,yCurrent] != self.destinationReward:
        actionIndex = self.nextAction(xCurrent,yCurrent,1)
        xNext,yNext = self.nextLocation(xCurrent,yCurrent,actionIndex)

        if [xNext,yNext] in obstacleArray:
          self.addObstacle(xNext,yNext)
          self.initializeQLearning(self.discountFactor,self.greedyPolicy,self.learningRate)
          self.modelTraining(intermediateTrainingEpisodes)
          continue
        elif self.rewardArray[xNext,yNext] == self.wallsReward:
          self.modelTraining(intermediateTrainingEpisodes)
          continue
        elif [xNext,yNext] == [xCurrent,yCurrent]:
          self.modelTraining(intermediateTrainingEpisodes)
          continue
        else:
          xCurrent,yCurrent = xNext,yNext
          coordinatePathArray.append([xCurrent,yCurrent])

      self.initializeRewardArray()
      self.QArray = self.np.copy(currentQArray)
      self.initializeDestination(self.currentDestination[0] + 1,self.currentDestination[1] + 1)

      return ['Destination Reached',coordinatePathArray]


    
#Frontend Methods_______________________________________________________________

  def formFilledSquare(self,figure,ix,iy,color):

    figure.add_scatter(fill = 'toself',
                       fillcolor = color,
                       line = dict(color = 'black'),
                       mode = 'lines',
                       showlegend = False,
                       x = [ix,ix + 1,ix + 1,ix,ix],
                       y = [iy,iy,iy + 1,iy + 1,iy])


  def formFilledSquareObject(self,ix,iy,color):

    return self.go.Scatter(fill = 'toself',
                           fillcolor = color,
                           line = dict(color = 'black'),
                           mode = 'lines',
                           showlegend = False,
                           x = [ix,ix + 1,ix + 1,ix,ix],
                           y = [iy,iy,iy + 1,iy + 1,iy])


  def formNumericGuides(self,figure,ix,iy,guideNumber):

    figure.add_scatter(mode = 'text',
                       showlegend = False,
                       text = guideNumber,
                       textposition = 'middle center',
                       x = [ix],
                       y = [iy])


  def formStaticMaze(self):

    staticMaze = self.go.Figure()

    staticMaze.update_layout(font = dict(family = 'Bahnschrift'),
                             grid = dict(columns = 1,
                                         rows = 1),
                             hovermode = False,
                             margin = dict(b = 0,
                                           l = 0,
                                           r = 0,
                                           t = 0),
                             paper_bgcolor = 'white',
                             plot_bgcolor = 'white',
                             showlegend = False,
                             xaxis = dict(visible = False),
                             yaxis = dict(scaleanchor = 'x',
                                          scaleratio = 1,
                                          visible = False))
    
    for ix in range(self.xColumns):
      for iy in range(self.yColumns):

        if self.rewardArray[ix,iy] == self.aislesReward:
          self.formFilledSquare(staticMaze,ix,iy,'white')
        elif self.rewardArray[ix,iy] == self.destinationReward:
          self.formFilledSquare(staticMaze,ix,iy,'lawngreen')
        elif self.rewardArray[ix,iy] == self.wallsReward:
          self.formFilledSquare(staticMaze,ix,iy,'darkslategrey')

    for ix in range(self.xColumns):
      self.formNumericGuides(staticMaze,ix + 0.5,- 0.5,ix + 1)
      self.formNumericGuides(staticMaze,ix + 0.5,self.yColumns + 0.5,ix + 1)

    for iy in range(self.yColumns):
      self.formNumericGuides(staticMaze,- 0.5,iy + 0.5,iy + 1)
      self.formNumericGuides(staticMaze,self.yColumns + 0.5,iy + 0.5,iy + 1)

    return staticMaze.show()


  def formDynamicMazeQValuesConstant(self,xStart,yStart,transitionDuration = 1000):

    coordinatePath = self.coordinatePathQValuesConstant(xStart,yStart)

    if coordinatePath[0] not in ['Initial Point Identical to the Destination Point','Initial Point Coinciding with the Maze Walls']:
      dynamicMaze = self.go.Figure()

      xShiftAnimatedMaze = self.xColumns + 1

      dynamicMaze.update_layout(font = dict(family = 'Bahnschrift'),
                                grid = dict(columns = 1,
                                            rows = 1),
                                hovermode = False,
                                margin = dict(b = 0,
                                              l = 0,
                                              r = 100,
                                              t = 0),
                                paper_bgcolor = 'white',
                                plot_bgcolor = 'white',
                                showlegend = False,
                                updatemenus = [dict(bgcolor = 'lightgrey',
                                                    bordercolor = 'darkgrey',
                                                    borderwidth = 1,
                                                    buttons = [dict(args = [None,
                                                                            dict(fromcurrent = True,
                                                                                frame = dict(duration = transitionDuration))],
                                                                    label = 'Play',
                                                                    method = 'animate'),
                                                              dict(args = [[None],
                                                                            dict(mode = 'immediate')],
                                                                    label = 'Pause',
                                                                    method = 'animate'),
                                                              dict(args = [None,
                                                                            dict(fromcurrent = False,
                                                                                frame = dict(duration = transitionDuration,
                                                                                              redraw = True))],
                                                                    label = 'Restart',
                                                                    method = 'animate')],
                                                    direction = 'up',
                                                    pad = dict(b = 0,
                                                               l = 10,
                                                               r = 0,
                                                               t = 0,),
                                                    showactive = True,
                                                    type = 'buttons',
                                                    x = 1,
                                                    xanchor = 'left',
                                                    y = 0.5,
                                                    yanchor = 'middle')],
                                xaxis = dict(visible = False),
                                yaxis = dict(scaleanchor = 'x',
                                            scaleratio = 1,
                                            visible = False))
      
      for i in range(3):
        for ix in range(1,len(coordinatePath[1]) - 1):
          self.formFilledSquare(dynamicMaze,coordinatePath[1][ix][0],coordinatePath[1][ix][1],'darkorange')

        if self.rewardArray[coordinatePath[1][- 1][0],coordinatePath[1][- 1][1]] == self.destinationReward:
          self.formFilledSquare(dynamicMaze,coordinatePath[1][- 1][0],coordinatePath[1][- 1][1],'lawngreen')
        else:
          self.formFilledSquare(dynamicMaze,coordinatePath[1][- 1][0],coordinatePath[1][- 1][1],'forestgreen')

        for ix in range(self.xColumns):
          for iy in range(self.yColumns):

            if [ix,iy] in coordinatePath[1]:
              continue
            else:
              if self.rewardArray[ix,iy] == self.aislesReward:
                self.formFilledSquare(dynamicMaze,ix,iy,'white')
              elif self.rewardArray[ix,iy] == self.destinationReward:
                self.formFilledSquare(dynamicMaze,ix,iy,'lawngreen')
              elif self.rewardArray[ix,iy] == self.wallsReward:
                self.formFilledSquare(dynamicMaze,ix,iy,'darkslategrey')

        self.formFilledSquare(dynamicMaze,coordinatePath[1][0][0],coordinatePath[1][0][1],'red')

      dynamicMaze.add_scatter(line = dict(color = 'black'),
                              mode = 'lines',
                              showlegend = False,
                              x = [xShiftAnimatedMaze,self.xColumns + xShiftAnimatedMaze,self.xColumns + xShiftAnimatedMaze,xShiftAnimatedMaze,xShiftAnimatedMaze],
                              y = [0,0,self.yColumns,self.yColumns,0])

      for ix in range(self.xColumns):
        self.formNumericGuides(dynamicMaze,ix + 0.5,- 0.5,ix + 1)
        self.formNumericGuides(dynamicMaze,ix + 0.5,self.yColumns + 0.5,ix + 1)
        self.formNumericGuides(dynamicMaze,ix + 0.5 + xShiftAnimatedMaze,- 0.5,ix + 1)
        self.formNumericGuides(dynamicMaze,ix + 0.5 + xShiftAnimatedMaze,self.yColumns + 0.5,ix + 1)

      for iy in range(self.yColumns):
        self.formNumericGuides(dynamicMaze,- 0.5,iy + 0.5,iy + 1)
        self.formNumericGuides(dynamicMaze,self.xColumns + 0.5,iy + 0.5,iy + 1)
        self.formNumericGuides(dynamicMaze,self.xColumns + 0.5 + xShiftAnimatedMaze,iy + 0.5,iy + 1)

      animatedMazeStaticScatterObject = []

      if self.rewardArray[coordinatePath[1][- 1][0],coordinatePath[1][- 1][1]] == self.destinationReward:
        animatedMazeStaticScatterObject.append(self.formFilledSquareObject(coordinatePath[1][- 1][0] + xShiftAnimatedMaze,coordinatePath[1][- 1][1],'lawngreen'))
      else:
        animatedMazeStaticScatterObject.append(self.formFilledSquareObject(coordinatePath[1][- 1][0] + xShiftAnimatedMaze,coordinatePath[1][- 1][1],'forestgreen'))

      for ix in range(self.xColumns):
        for iy in range(self.yColumns):

          if [ix,iy] in coordinatePath[1]:
            continue
          else:
            if self.rewardArray[ix,iy] == self.aislesReward:
              animatedMazeStaticScatterObject.append(self.formFilledSquareObject(ix + xShiftAnimatedMaze,iy,'white'))
            elif self.rewardArray[ix,iy] == self.destinationReward:
              animatedMazeStaticScatterObject.append(self.formFilledSquareObject(ix + xShiftAnimatedMaze,iy,'lawngreen'))
            elif self.rewardArray[ix,iy] == self.wallsReward:
              animatedMazeStaticScatterObject.append(self.formFilledSquareObject(ix + xShiftAnimatedMaze,iy,'darkslategrey'))

      animatedMazeStaticScatterObject.append(self.formFilledSquareObject(coordinatePath[1][0][0] + xShiftAnimatedMaze,coordinatePath[1][0][1],'red'))

      coordinatePath[1].pop(0)
      coordinatePath[1].pop(-1)

      dynamicMaze.frames = tuple(self.go.Frame(data = animatedMazeStaticScatterObject + [self.formFilledSquareObject(ixy[0] + xShiftAnimatedMaze,ixy[1],'darkorange')]) for ixy in coordinatePath[1])

      return dynamicMaze.show()
    else:
      return print(coordinatePath[0])


  def formDynamicMazeQValuesUpdating(self,xStart,yStart,obstacleArray,intermediateTrainingEpisodes,transitionDuration = 1000):

    obstacleArray = [list(i) for i in self.np.array(obstacleArray) - 1] #obstacleArray: Transformation of obstacleArray(exluding 0) to 2-Dimensional array consists of coordinates of obstacle cells in maze(Including 0)
    coordinatePath = self.coordinatePathQValuesUpdating(xStart,yStart,obstacleArray,intermediateTrainingEpisodes)

    if coordinatePath[0] not in ['Initial Point Identical to the Destination Point','Initial Point Coinciding with the Maze Walls','Initial Point Coinciding with a Obstacle Point']:
      dynamicMaze = self.go.Figure()

      xShiftAnimatedMaze = self.xColumns + 1

      dynamicMaze.update_layout(font = dict(family = 'Bahnschrift'),
                                grid = dict(columns = 1,
                                            rows = 1),
                                hovermode = False,
                                margin = dict(b = 0,
                                              l = 0,
                                              r = 100,
                                              t = 0),
                                paper_bgcolor = 'white',
                                plot_bgcolor = 'white',
                                showlegend = False,
                                updatemenus = [dict(bgcolor = 'lightgrey',
                                                    bordercolor = 'darkgrey',
                                                    borderwidth = 1,
                                                    buttons = [dict(args = [None,
                                                                            dict(fromcurrent = True,
                                                                                frame = dict(duration = transitionDuration))],
                                                                    label = 'Play',
                                                                    method = 'animate'),
                                                              dict(args = [[None],
                                                                            dict(mode = 'immediate')],
                                                                    label = 'Pause',
                                                                    method = 'animate'),
                                                              dict(args = [None,
                                                                            dict(fromcurrent = False,
                                                                                frame = dict(duration = transitionDuration,
                                                                                              redraw = True))],
                                                                    label = 'Restart',
                                                                    method = 'animate')],
                                                    direction = 'up',
                                                    pad = dict(b = 0,
                                                               l = 10,
                                                               r = 0,
                                                               t = 0,),
                                                    showactive = True,
                                                    type = 'buttons',
                                                    x = 1,
                                                    xanchor = 'left',
                                                    y = 0.5,
                                                    yanchor = 'middle')],
                                xaxis = dict(visible = False),
                                yaxis = dict(scaleanchor = 'x',
                                            scaleratio = 1,
                                            visible = False))
      
      for i in range(3):
        for ix in range(1,len(coordinatePath[1]) - 1):
          self.formFilledSquare(dynamicMaze,coordinatePath[1][ix][0],coordinatePath[1][ix][1],'darkorange')

        if self.rewardArray[coordinatePath[1][- 1][0],coordinatePath[1][- 1][1]] == self.destinationReward:
          self.formFilledSquare(dynamicMaze,coordinatePath[1][- 1][0],coordinatePath[1][- 1][1],'lawngreen')
        else:
          self.formFilledSquare(dynamicMaze,coordinatePath[1][- 1][0],coordinatePath[1][- 1][1],'forestgreen')

        for ix in range(self.xColumns):
          for iy in range(self.yColumns):

            if [ix,iy] in coordinatePath[1]:
              continue
            else:
              if self.rewardArray[ix,iy] == self.aislesReward:
                if [ix,iy] in obstacleArray:
                  self.formFilledSquare(dynamicMaze,ix,iy,'mediumspringgreen')
                else:
                  self.formFilledSquare(dynamicMaze,ix,iy,'white')
              elif self.rewardArray[ix,iy] == self.destinationReward:
                self.formFilledSquare(dynamicMaze,ix,iy,'lawngreen')
              elif self.rewardArray[ix,iy] == self.wallsReward:
                self.formFilledSquare(dynamicMaze,ix,iy,'darkslategrey')

        self.formFilledSquare(dynamicMaze,coordinatePath[1][0][0],coordinatePath[1][0][1],'red')

      dynamicMaze.add_scatter(line = dict(color = 'black'),
                              mode = 'lines',
                              showlegend = False,
                              x = [xShiftAnimatedMaze,self.xColumns + xShiftAnimatedMaze,self.xColumns + xShiftAnimatedMaze,xShiftAnimatedMaze,xShiftAnimatedMaze],
                              y = [0,0,self.yColumns,self.yColumns,0])

      for ix in range(self.xColumns):
        self.formNumericGuides(dynamicMaze,ix + 0.5,- 0.5,ix + 1)
        self.formNumericGuides(dynamicMaze,ix + 0.5,self.yColumns + 0.5,ix + 1)
        self.formNumericGuides(dynamicMaze,ix + 0.5 + xShiftAnimatedMaze,- 0.5,ix + 1)
        self.formNumericGuides(dynamicMaze,ix + 0.5 + xShiftAnimatedMaze,self.yColumns + 0.5,ix + 1)

      for iy in range(self.yColumns):
        self.formNumericGuides(dynamicMaze,- 0.5,iy + 0.5,iy + 1)
        self.formNumericGuides(dynamicMaze,self.xColumns + 0.5,iy + 0.5,iy + 1)
        self.formNumericGuides(dynamicMaze,self.xColumns + 0.5 + xShiftAnimatedMaze,iy + 0.5,iy + 1)

      animatedMazeStaticScatterObject = []

      animatedMazeStaticScatterObject.append(self.formFilledSquareObject(coordinatePath[1][0][0] + xShiftAnimatedMaze,coordinatePath[1][0][1],'red'))

      if self.rewardArray[coordinatePath[1][- 1][0],coordinatePath[1][- 1][1]] == self.destinationReward:
        animatedMazeStaticScatterObject.append(self.formFilledSquareObject(coordinatePath[1][- 1][0] + xShiftAnimatedMaze,coordinatePath[1][- 1][1],'lawngreen'))
      else:
        animatedMazeStaticScatterObject.append(self.formFilledSquareObject(coordinatePath[1][- 1][0] + xShiftAnimatedMaze,coordinatePath[1][- 1][1],'forestgreen'))

      for ix in range(self.xColumns):
        for iy in range(self.yColumns):

          if [ix,iy] in coordinatePath[1]:
            continue
          else:
            if self.rewardArray[ix,iy] == self.aislesReward:
              if [ix,iy] in obstacleArray:
                animatedMazeStaticScatterObject.append(self.formFilledSquareObject(ix + xShiftAnimatedMaze,iy,'mediumspringgreen'))
              else:
                animatedMazeStaticScatterObject.append(self.formFilledSquareObject(ix + xShiftAnimatedMaze,iy,'white'))
            elif self.rewardArray[ix,iy] == self.destinationReward:
              animatedMazeStaticScatterObject.append(self.formFilledSquareObject(ix + xShiftAnimatedMaze,iy,'lawngreen'))
            elif self.rewardArray[ix,iy] == self.wallsReward:
              animatedMazeStaticScatterObject.append(self.formFilledSquareObject(ix + xShiftAnimatedMaze,iy,'darkslategrey'))

      coordinatePath[1].pop(0)
      coordinatePath[1].pop(-1)

      dynamicMaze.frames = tuple(self.go.Frame(data = animatedMazeStaticScatterObject + [self.formFilledSquareObject(ixy[0] + xShiftAnimatedMaze,ixy[1],'darkorange')]) for ixy in coordinatePath[1])

      return dynamicMaze.show()
    else:
      return print(coordinatePath[0])
