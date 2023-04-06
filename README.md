<!DOCTYPE html>
<html>
<head>
</head>
<body>
	<h1>Maze Solving Application</h1>

	<h2>Overview</h2>

	<p>The application allows users to define a maze and a destination within the maze, and then train an algorithm to find the shortest path to the destination. Users can specify the algorithm parameters, such as the discount factor, greedy policy, and learning rate. They can also simulate the algorithm with and without obstacles in the maze.</p>

	<h2>Dependencies</h2>

	<p>This project requires the following dependencies:</p>

	<ul>
		<li>Python 3.6 or higher</li>
		<li>Plotly</li>
	</ul>

	<p>You can install the Plotly library using pip:</p>

	<pre><code>pip install plotly</code></pre>

	<h2>Usage</h2>

	<p>To use this project, follow these steps:</p>

	<ol>
		<li>Define the maze using the <code>maze.py</code> module. You can specify the size of the maze and the location of the walls.</li>
		<li>Define the destination using the <code>maze.py</code> module. You can specify the location of the destination within the maze.</li>
		<li>Define the algorithm parameters using the <code>algorithm.py</code> module. You can specify the discount factor, greedy policy, and learning rate.</li>
		<li>Train the algorithm using the <code>train.py</code> module. This will create a Q-table for the algorithm based on the specified maze, destination, and algorithm parameters.</li>
		<li>Define the initial point using the <code>simulation.py</code> module. You can specify the starting point for the algorithm within the maze.</li>
		<li>Simulate the algorithm without obstacles using the <code>simulation.py</code> module. This will show the shortest path to the destination without any obstacles in the maze.</li>
		<li>Define obstacles using the <code>maze.py</code> module. You can specify the location of the obstacles within the maze.</li>
		<li>Simulate the algorithm with obstacles using the <code>simulation.py</code> module. This will show the shortest path to the destination with the specified obstacles in the maze.</li>
	</ol>

	<h2>Credits</h2>

	<p>This project was created by Dinith Heshan as a part of his final year project for his engineering degree.</p>

	<h2>License</h2>

	<p>This project is licensed under the MIT License. See the LICENSE file for details.</p>
</body>
</html>
