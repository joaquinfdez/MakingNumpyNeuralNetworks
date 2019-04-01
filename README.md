<!DOCTYPE html>
<html>
<body>
  <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">

<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<div style="width: 100%; clear: both;">
<div style="float: left; width: 50%;">
<img src="http://www.uoc.edu/portal/_resources/common/imatges/marca_UOC/UOC_Masterbrand.jpg", align="left">
</div>
<div style="float: right; width: 50%;">
<p style="margin: 0; padding-top: 22px; text-align:right;">M2.875 · Deep Learning · PEC1</p>
<p style="margin: 0; text-align:right;">2018-2 · Máster universitario en Ciencia de datos (Data science)</p>
<p style="margin: 0; text-align:right; padding-button: 100px;">Estudios de Informática, Multimedia y Telecomunicación</p>
</div>
</div>
<div style="width:100%;">&nbsp;</div><h1 id="PEC-1:-Redes-neuronales-completamente-conectadas">PEC 1: Redes neuronales completamente conectadas<a class="anchor-link" href="#PEC-1:-Redes-neuronales-completamente-conectadas">&#182;</a></h1><p>En esta práctica implementaremos una red neuronal completamente conectada de dos formas diferentes:</p>
<ol start="1">
  <li>Partiendo de cero utilizando únicamente la librería numpy</li>
  <li>Utilizando la librería Keras y TensorFlow</li>
</ol><p>Posteriormente utilizaremos las dos implementaciones para entrenar dos redes neuronales iguales en un conjunto de datos y compararemos el rendimiento.</p>
<p><strong>Importante: Cada uno de los ejercicios puede suponer varios minutos de ejecución, por lo que la entrega debe hacerse en formato notebook y en formato html donde se vea el código y los resultados y comentarios de cada ejercicio. Para exportar el notebook a html puede hacerse desde el menú File $\to$ Download as $\to$ HTML.</strong></p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="0.-Carga-de-datos">0. Carga de datos<a class="anchor-link" href="#0.-Carga-de-datos">&#182;</a></h2><p>El siguiente código carga los paquetes necesarios para la práctica y además lee los datos que utilizaremos para entrenar la red neuronal.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">pickle</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="k">import</span> <span class="n">train_test_split</span>

<span class="k">with</span> <span class="nb">open</span><span class="p">(</span><span class="s2">&quot;data.pickle&quot;</span><span class="p">,</span> <span class="s2">&quot;rb&quot;</span><span class="p">)</span> <span class="k">as</span> <span class="n">f</span><span class="p">:</span>
    <span class="n">data</span> <span class="o">=</span> <span class="n">pickle</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="n">f</span><span class="p">)</span>

<span class="n">features</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="s2">&quot;features&quot;</span><span class="p">]</span>
<span class="n">labels</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="s2">&quot;labels&quot;</span><span class="p">]</span>

<span class="n">train_x</span><span class="p">,</span> <span class="n">test_x</span><span class="p">,</span> <span class="n">train_y</span><span class="p">,</span> <span class="n">test_y</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">features</span><span class="p">,</span> <span class="n">labels</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.2</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="1.-Redes-neuronales-utilizando-numpy">1. Redes neuronales utilizando numpy<a class="anchor-link" href="#1.-Redes-neuronales-utilizando-numpy">&#182;</a></h2><p>A continuación implementaremos todas las funciones necesarias para entrenar una red neuronal completamente conectada utilizando únicamente la librería numpy. El objetivo es poder entrenar una red neuronal con cualquier número de capas en la cual la última capa tendrá una única neurona con función de activación sigmoid y las demás capas cualquier número de neuronas con función de activación relu.</p>
<p>La siguiente figura muestra un diagrama de como implementaremos el proceso de entrenamiento de la red neuronal:</p>
<p><img src="diag.png" alt="Diagrama del entrenamiento de la red neuronal" style="height: 550px;"/></p>
<p>El desarrollo está estructurado en funciones básicas que se componen según el siguiente esquema:</p>
<ul>
<li>L_layer_model<ul>
<li>initialize_parameters</li>
<li>L_model_forward<ul>
<li>linear_activation_forward<ul>
<li>linear_forward</li>
<li>sigmoid</li>
<li>relu</li>
</ul>
</li>
</ul>
</li>
<li>compute_cost</li>
<li>L_model_backward<ul>
<li>linear_activation_backward<ul>
<li>linear_backward</li>
<li>sigmoid_backward</li>
<li>relu_backward</li>
</ul>
</li>
</ul>
</li>
<li>update_parameters</li>
</ul>
</li>
<li>accuracy</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><strong>Notación</strong>:</p>
<ul>
<li>Denotamos $L$ el número de capas de la red neuronal.</li>
<li>La matriz de pesos que conecta una capa con la siguiente la denotamos con la letra $W$, mientras que el vector de bias lo denotamos con la letra $b$.</li>
<li>Superíndice $[l]$ denota una cantidad asociada con la capa número $l$. <ul>
<li>Ejemplo: $a^{[L]}$ denota la salida de la capa número $L$.</li>
<li>Ejemplo: Las variables $W^{[L]}$ y $b^{[L]}$ denotan la matriz de pesos y el vector de bias que conectan la capa $L-1$ con la capa $L$ respectivamente.</li>
</ul>
</li>
<li>Superíndice $(i)$ denota una cantidad asociada con el ejemplo $i$-ésimo. <ul>
<li>Ejemplo: $x^{(i)}$ es el ejemplo del conjunto de entrenamiento número $i$.</li>
</ul>
</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="1.1-Inicializaci&#243;n-de-par&#225;metros">1.1 Inicializaci&#243;n de par&#225;metros<a class="anchor-link" href="#1.1-Inicializaci&#243;n-de-par&#225;metros">&#182;</a></h3><p>El primer paso para entrenar una red neuronal consiste en inicializar de forma aleatoria los parámetros del modelo.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Ejercicio:</strong> Inicializar las matrices de parámetros y los vectores de bias. Las matrices de pesos se deben inicializar utilizando la distribución normal y los vectores de bias se deben inicializar con ceros. Para las matrices de pesos podéis utilizar 0.1*np.random.randn(shape) indicando el tamaño correcto.
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">initialize_parameters</span><span class="p">(</span><span class="n">layer_dims</span><span class="p">):</span>
    <span class="sd">&quot;&quot;&quot;</span>
<span class="sd">    Argumentos:</span>
<span class="sd">    layer_dims -- lista que contiene las dimensiones de cada capa de la red</span>
<span class="sd">    </span>
<span class="sd">    Devuelve:</span>
<span class="sd">    parameters -- diccionario que contiene los parametros &quot;W1&quot;, &quot;b1&quot;, ..., &quot;WL&quot;, &quot;bL&quot;:</span>
<span class="sd">                    Wl -- matriz de pesos de tamaño (layer_dims[l], layer_dims[l-1])</span>
<span class="sd">                    bl -- vector de bias de tamaño (layer_dims[l], 1)</span>
<span class="sd">    &quot;&quot;&quot;</span>
    
    <span class="n">parameters</span> <span class="o">=</span> <span class="p">{}</span>
    <span class="n">L</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">layer_dims</span><span class="p">)</span>
    <span class="c1">#print(&quot;Nº de parametros: &quot;, L)</span>
    
    <span class="k">for</span> <span class="n">l</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="n">L</span><span class="p">):</span>
        <span class="c1">#print(&quot;Capa de entrada: &quot;, layer_dims[l-1], &quot;capa de salida&quot;, layer_dims[l])</span>
        <span class="n">parameters</span><span class="p">[</span><span class="s1">&#39;W&#39;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">l</span><span class="p">)]</span> <span class="o">=</span> <span class="mf">0.1</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span><span class="n">layer_dims</span><span class="p">[</span><span class="n">l</span><span class="p">],</span> <span class="n">layer_dims</span><span class="p">[</span><span class="n">l</span><span class="o">-</span><span class="mi">1</span><span class="p">])</span>
        <span class="n">parameters</span><span class="p">[</span><span class="s1">&#39;b&#39;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">l</span><span class="p">)]</span> <span class="o">=</span> <span class="mf">0.1</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span> <span class="n">layer_dims</span><span class="p">[</span><span class="n">l</span><span class="p">],</span> <span class="mi">1</span><span class="p">)</span>
    
    <span class="k">return</span> <span class="n">parameters</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="1.2-Propagaci&#243;n-hacia-delante">1.2 Propagaci&#243;n hacia delante<a class="anchor-link" href="#1.2-Propagaci&#243;n-hacia-delante">&#182;</a></h3><p>En una capa concreta de una red neuronal, las entradas de las neuronas se combinan de forma lineal antes de pasar por la función de activación según la siguiente fórmula:</p>
$$Z^{[l]} = W^{[l]}A^{[l-1]} +b^{[l]}$$
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Ejercicio:</strong> Calcular la combinación lineal de las entradas a una capa de la red neuronal.
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">linear_forward</span><span class="p">(</span><span class="n">A</span><span class="p">,</span> <span class="n">W</span><span class="p">,</span> <span class="n">b</span><span class="p">):</span>
    <span class="sd">&quot;&quot;&quot;</span>
<span class="sd">    Implementa la parte lineal de la propagación hacia delante de una capa</span>

<span class="sd">    Argumentos:</span>
<span class="sd">    A -- salida de la capa anterior (o datos de entrada): (número de neuronas de la capa anterior, número de ejemplos)</span>
<span class="sd">    W -- matriz de pesos: (número de neuronas de la capa actual, número de neuronas de la capa anterior)</span>
<span class="sd">    b -- vector de bias: (número de neuronas de la capa actual, 1)</span>

<span class="sd">    Devuelve:</span>
<span class="sd">    Z -- la entrada a la función de activación</span>
<span class="sd">    cache -- una tripleta que contiene &quot;A&quot;, &quot;W&quot; y &quot;b&quot;, utilizada después para la propagación hacia atrás</span>
<span class="sd">    &quot;&quot;&quot;</span>
    
    <span class="n">Z</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">dot</span><span class="p">(</span><span class="n">W</span><span class="p">,</span> <span class="n">A</span><span class="p">)</span> <span class="o">+</span> <span class="n">b</span>
    <span class="n">cache</span> <span class="o">=</span> <span class="p">(</span><span class="n">A</span><span class="p">,</span> <span class="n">W</span><span class="p">,</span> <span class="n">b</span><span class="p">)</span>
    <span class="c1">#print(&quot;W shape&quot;, W.shape, &quot; A shape &quot;, A.shape, &quot; b shape&quot;, b.shape)</span>
    <span class="c1">#print(&quot;linear_forward; Z=&quot;, Z)</span>
    <span class="k">return</span> <span class="n">Z</span><span class="p">,</span> <span class="n">cache</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Una vez se ha calculado la combinación lineal de las entradas de una capa se debe aplicar una función de activación no lineal antes de enviar las salidas a la siguiente capa. Si denotamos $g$ la función de activación (en nuestro caso relu o sigmoid), tenemos la siguiente fórmula:</p>
$$A^{[l]} = g(Z^{[l]}) = g(W^{[l]}A^{[l-1]} + b^{[l]})$$
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>A continuación definimos las funciones de activación que utilizaremos en la red neuronal.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">sigmoid</span><span class="p">(</span><span class="n">Z</span><span class="p">):</span>
    <span class="n">A</span> <span class="o">=</span> <span class="mi">1</span> <span class="o">/</span> <span class="p">(</span><span class="mi">1</span> <span class="o">+</span> <span class="n">np</span><span class="o">.</span><span class="n">exp</span><span class="p">(</span><span class="o">-</span><span class="n">Z</span><span class="p">))</span>
    <span class="n">cache</span> <span class="o">=</span> <span class="n">Z</span>
    <span class="k">return</span> <span class="n">A</span><span class="p">,</span> <span class="n">cache</span>

<span class="k">def</span> <span class="nf">relu</span><span class="p">(</span><span class="n">Z</span><span class="p">):</span>
    <span class="n">A</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">maximum</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="n">Z</span><span class="p">)</span>
    <span class="n">cache</span> <span class="o">=</span> <span class="n">Z</span>
    <span class="k">return</span> <span class="n">A</span><span class="p">,</span> <span class="n">cache</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Ejercicio:</strong> Calcular la combinación lineal de las entradas utilizando la función implementada anteriormente y aplicar la función de activación no lineal que corresponda.
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">linear_activation_forward</span><span class="p">(</span><span class="n">A_prev</span><span class="p">,</span> <span class="n">W</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">activation</span><span class="p">):</span>
    <span class="sd">&quot;&quot;&quot;</span>
<span class="sd">    Implementa la propagación hacia delante de una capa incluyendo la función de activación</span>

<span class="sd">    Argumentos:</span>
<span class="sd">    A_prev -- salida de la capa anterior (o datos de entrada): </span>
<span class="sd">                (número de neuronas de la capa anterior, número de ejemplos)</span>
<span class="sd">    W -- matriz de pesos: (número de neuronas de la capa actual, número de neuronas de la capa anterior)</span>
<span class="sd">    b -- vector de bias: (número de neuronas de la capa actual, 1)</span>
<span class="sd">    activation -- el nombre de la función de activación a utilizar en la capa: &quot;sigmoid&quot; o &quot;relu&quot;</span>

<span class="sd">    Devuelve:</span>
<span class="sd">    A -- la salida de la capa después de aplicar la función de activación</span>
<span class="sd">    cache -- una dupla que contiene &quot;linear_cache&quot; y &quot;activation_cache&quot;, utilizada después para la propagación hacia atrás</span>
<span class="sd">    &quot;&quot;&quot;</span>
    
    <span class="n">Z</span><span class="p">,</span> <span class="n">linear_cache</span> <span class="o">=</span> <span class="n">linear_forward</span><span class="p">(</span><span class="n">A_prev</span><span class="p">,</span> <span class="n">W</span><span class="p">,</span> <span class="n">b</span><span class="p">)</span>
    
    <span class="k">if</span> <span class="n">activation</span> <span class="o">==</span> <span class="s2">&quot;sigmoid&quot;</span><span class="p">:</span>    
        <span class="n">A</span><span class="p">,</span> <span class="n">activation_cache</span> <span class="o">=</span> <span class="n">sigmoid</span><span class="p">(</span><span class="n">Z</span><span class="p">)</span>
    <span class="k">elif</span> <span class="n">activation</span> <span class="o">==</span> <span class="s2">&quot;relu&quot;</span><span class="p">:</span>
        <span class="n">A</span><span class="p">,</span> <span class="n">activation_cache</span> <span class="o">=</span> <span class="n">relu</span><span class="p">(</span><span class="n">Z</span><span class="p">)</span>
        
    <span class="n">cache</span> <span class="o">=</span> <span class="p">(</span><span class="n">linear_cache</span><span class="p">,</span> <span class="n">activation_cache</span><span class="p">)</span>

    <span class="k">return</span> <span class="n">A</span><span class="p">,</span> <span class="n">cache</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Dados los datos de entrada, la salida de la red neuronal se calcula aplicando diferentes capas una detrás de otra. Si denotamos la última capa como $L$, la salida de la red neuronal se corresponde con la salida de la última capa $A^{[L]}$.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Ejercicio:</strong> Calcular la salida de la red neuronal aplicando $L-1$ capas con función de activación relu y una última capa con función de activación sigmoid.
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">L_model_forward</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">parameters</span><span class="p">):</span>
    <span class="sd">&quot;&quot;&quot;</span>
<span class="sd">    Implementa la propagación hacia delante de la red neuronal completa</span>
<span class="sd">    </span>
<span class="sd">    Argumentos:</span>
<span class="sd">    X -- datos: matriz de tamaño (número de variables, número de ejemplos)</span>
<span class="sd">    parameters -- salida de la función initialize_parameters()</span>
<span class="sd">    </span>
<span class="sd">    Devuelve:</span>
<span class="sd">    AL -- salida de la red neuronal</span>
<span class="sd">    caches -- lista de caches que contiene todas las caches de la función linear_activation_forward(), las caches</span>
<span class="sd">                indexadas de 0 a L-2 corresponden a las caches de la función de activación relu y la cache indexada</span>
<span class="sd">                como L-1 corresponde a la cache de la función de activación sigmoid</span>
<span class="sd">    &quot;&quot;&quot;</span>

    <span class="n">caches</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="n">A</span> <span class="o">=</span> <span class="n">X</span>
    <span class="n">L</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">parameters</span><span class="p">)</span> <span class="o">//</span> <span class="mi">2</span>
    
    <span class="c1"># Implementa primero las L-1 capas con función de activación relu</span>
    <span class="k">for</span> <span class="n">l</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="n">L</span><span class="p">):</span>
        <span class="n">A_prev</span> <span class="o">=</span> <span class="n">A</span> 
        
        <span class="n">activation</span> <span class="o">=</span> <span class="s2">&quot;relu&quot;</span>
        <span class="n">W_current</span> <span class="o">=</span> <span class="n">parameters</span><span class="p">[</span><span class="s1">&#39;W&#39;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">l</span><span class="p">)]</span>
        <span class="n">b_current</span> <span class="o">=</span> <span class="n">parameters</span><span class="p">[</span><span class="s1">&#39;b&#39;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">l</span><span class="p">)]</span>
        
        <span class="n">A</span><span class="p">,</span> <span class="n">cache</span> <span class="o">=</span> <span class="n">linear_activation_forward</span><span class="p">(</span><span class="n">A_prev</span><span class="p">,</span> <span class="n">W_current</span><span class="p">,</span> <span class="n">b_current</span><span class="p">,</span> <span class="n">activation</span><span class="p">)</span>
        
        <span class="n">caches</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">cache</span><span class="p">)</span>
    
    <span class="c1"># Implementa la última capa con función de activación sigmoid</span>
    <span class="n">activation</span> <span class="o">=</span> <span class="s2">&quot;sigmoid&quot;</span>
    <span class="n">W_current</span> <span class="o">=</span> <span class="n">parameters</span><span class="p">[</span><span class="s1">&#39;W&#39;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">L</span><span class="p">)]</span>
    <span class="n">b_current</span> <span class="o">=</span> <span class="n">parameters</span><span class="p">[</span><span class="s1">&#39;b&#39;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">L</span><span class="p">)]</span>
    <span class="n">AL</span><span class="p">,</span> <span class="n">cache</span> <span class="o">=</span> <span class="n">linear_activation_forward</span><span class="p">(</span><span class="n">A</span><span class="p">,</span> <span class="n">W_current</span><span class="p">,</span> <span class="n">b_current</span><span class="p">,</span> <span class="n">activation</span><span class="p">)</span>
    <span class="n">caches</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">cache</span><span class="p">)</span>
    
    <span class="k">return</span> <span class="n">AL</span><span class="p">,</span> <span class="n">caches</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="1.3-Funci&#243;n-de-coste">1.3 Funci&#243;n de coste<a class="anchor-link" href="#1.3-Funci&#243;n-de-coste">&#182;</a></h3><p>Una vez hemos obtenido la salida de la red neuronal podemos obtener un valor que mida el rendimiento de la red neuronal utilizando una función de coste $\mathcal{L}$. En nuestro caso utilizaremos la función de coste log-loss, que viene definida por la siguiente fórmula:</p>
$$\mathcal{L} = -\frac{1}{m} \sum\limits_{i = 1}^{m} (y^{(i)}\log\left(a^{[L] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right))$$
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Ejercicio:</strong> Calcular el valor de la función de coste log-loss dada la salida de la red neuronal junto con las etiquetas correctas.
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">compute_cost</span><span class="p">(</span><span class="n">AL</span><span class="p">,</span> <span class="n">Y</span><span class="p">):</span>
    <span class="sd">&quot;&quot;&quot;</span>
<span class="sd">    Calcula la función de coste</span>

<span class="sd">    Argumentos:</span>
<span class="sd">    AL -- vector que contiene la salida de la red, corresponde a las probabilidades que predice la red neuronal</span>
<span class="sd">            para cada ejemplo: (1, número de ejemplos)</span>
<span class="sd">    Y -- vector con las etiquetas correctas para los datos de entrada a la red: (1, número de ejemplos)</span>

<span class="sd">    Devuelve:</span>
<span class="sd">    cost -- valor de la función de coste log-loss</span>
<span class="sd">    &quot;&quot;&quot;</span>
    
    <span class="n">m</span> <span class="o">=</span> <span class="n">Y</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span>

    <span class="n">cost</span> <span class="o">=</span> <span class="o">-</span><span class="mi">1</span> <span class="o">/</span> <span class="n">m</span> <span class="o">*</span> <span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">dot</span><span class="p">(</span><span class="n">Y</span><span class="p">,</span> <span class="n">np</span><span class="o">.</span><span class="n">log</span><span class="p">(</span><span class="n">AL</span><span class="p">)</span><span class="o">.</span><span class="n">T</span><span class="p">)</span> <span class="o">+</span> <span class="n">np</span><span class="o">.</span><span class="n">dot</span><span class="p">(</span><span class="mi">1</span> <span class="o">-</span> <span class="n">Y</span><span class="p">,</span> <span class="n">np</span><span class="o">.</span><span class="n">log</span><span class="p">(</span><span class="mi">1</span> <span class="o">-</span> <span class="n">AL</span><span class="p">)</span><span class="o">.</span><span class="n">T</span><span class="p">))</span>
    <span class="n">cost</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">squeeze</span><span class="p">(</span><span class="n">cost</span><span class="p">)</span>
    
    <span class="k">return</span> <span class="n">cost</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="1.4-Propagaci&#243;n-hacia-atr&#225;s">1.4 Propagaci&#243;n hacia atr&#225;s<a class="anchor-link" href="#1.4-Propagaci&#243;n-hacia-atr&#225;s">&#182;</a></h3><p>Para entrenar una red neuronal es necesario calcular el gradiente de la función de coste repescto a los parámetros de la red, para lo cual utilizaremos la propagación hacia atrás. La propagación hacia atrás consiste en aplicar la regla de la cadena para calcular el gradiente de la función de coste paso a paso en cada capa.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Para aplicar la regla de la cadena en la parte lineal de la neurona, supongamos que ya hemos calculado la derivada $dZ^{[l]} = \frac{\partial \mathcal{L} }{\partial Z^{[l]}}$. Entonces, para calcular las derivadas $(dW^{[l]}, db^{[l]}, dA^{[l-1]})$ podemos utilizar las siguientes fórmulas:</p>
$$ dW^{[l]} = \frac{\partial \mathcal{L} }{\partial W^{[l]}} = \frac{1}{m} dZ^{[l]} A^{[l-1] T}$$$$ db^{[l]} = \frac{\partial \mathcal{L} }{\partial b^{[l]}} = \frac{1}{m} \sum_{i = 1}^{m} dZ^{[l](i)}$$$$ dA^{[l-1]} = \frac{\partial \mathcal{L} }{\partial A^{[l-1]}} = W^{[l] T} dZ^{[l]}$$
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Ejercicio:</strong> Calcular las derivadas de la parte lineal para una sola capa.
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">linear_backward</span><span class="p">(</span><span class="n">dZ</span><span class="p">,</span> <span class="n">cache</span><span class="p">):</span>
    <span class="sd">&quot;&quot;&quot;</span>
<span class="sd">    Implementa la parte lineal de la propagación hacia atrás para una única capa</span>

<span class="sd">    Argumentos:</span>
<span class="sd">    dZ -- derivada de la función de coste con respecto a la salida lineal de la capa actual</span>
<span class="sd">    cache -- tripleta que contiene los valores (A_prev, W, b), provinientes de la función linear_forward</span>

<span class="sd">    Devuelve:</span>
<span class="sd">    dA_prev -- derivada de la función de coste con respecto a la salida de la capa anterior (l-1): </span>
<span class="sd">                tiene el mismo tamaño que A_prev</span>
<span class="sd">    dW -- derivada de la función de coste con respecto a la matriz de pesos W de la capa actual (l):</span>
<span class="sd">                tiene el mismo tamaño que W</span>
<span class="sd">    db -- derivada de la función de coste con respecto al vector de bias b de la capa actual (l):</span>
<span class="sd">                tiene el mismo tamaño que b</span>
<span class="sd">    &quot;&quot;&quot;</span>
    
    <span class="n">A_prev</span><span class="p">,</span> <span class="n">W</span><span class="p">,</span> <span class="n">b</span> <span class="o">=</span> <span class="n">cache</span>
    <span class="n">m</span> <span class="o">=</span> <span class="n">A_prev</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span>

    <span class="n">dW</span> <span class="o">=</span> <span class="mi">1</span> <span class="o">/</span> <span class="n">m</span> <span class="o">*</span> <span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">dot</span><span class="p">(</span><span class="n">dZ</span><span class="p">,</span> <span class="n">A_prev</span><span class="o">.</span><span class="n">T</span><span class="p">))</span>
    <span class="n">db</span> <span class="o">=</span> <span class="mi">1</span> <span class="o">/</span> <span class="n">m</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">sum</span><span class="p">(</span><span class="n">dZ</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">keepdims</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
    <span class="n">dA_prev</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">dot</span><span class="p">(</span><span class="n">W</span><span class="o">.</span><span class="n">T</span><span class="p">,</span> <span class="n">dZ</span><span class="p">)</span>

    <span class="k">return</span> <span class="n">dA_prev</span><span class="p">,</span> <span class="n">dW</span><span class="p">,</span> <span class="n">db</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>El siguiente paso consiste en aplicar la regla de la cadena a la parte no lineal de las neuronas, es decir, a las funciones de activación. Para esto, si denotamos $g$ la función de activación, podemos utilizar la siguiente fórmula:</p>
$$dZ^{[l]} = dA^{[l]} * g'(Z^{[l]})$$<p>Donde $*$ indica el producto componente a componente.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>A continuación calculamos las derivadas de las funciones de activación que utilizamos en la red neuronal.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">sigmoid_backward</span><span class="p">(</span><span class="n">dA</span><span class="p">,</span> <span class="n">cache</span><span class="p">):</span>
    <span class="n">Z</span> <span class="o">=</span> <span class="n">cache</span>
    <span class="n">s</span> <span class="o">=</span> <span class="mi">1</span> <span class="o">/</span> <span class="p">(</span><span class="mi">1</span> <span class="o">+</span> <span class="n">np</span><span class="o">.</span><span class="n">exp</span><span class="p">(</span><span class="o">-</span><span class="n">Z</span><span class="p">))</span>
    <span class="n">dZ</span> <span class="o">=</span> <span class="n">dA</span> <span class="o">*</span> <span class="n">s</span> <span class="o">*</span> <span class="p">(</span><span class="mi">1</span> <span class="o">-</span> <span class="n">s</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">dZ</span>

<span class="k">def</span> <span class="nf">relu_backward</span><span class="p">(</span><span class="n">dA</span><span class="p">,</span> <span class="n">cache</span><span class="p">):</span>
    <span class="n">Z</span> <span class="o">=</span> <span class="n">cache</span>
    <span class="n">dZ</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">dA</span><span class="p">,</span> <span class="n">copy</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
    <span class="n">dZ</span><span class="p">[</span><span class="n">Z</span> <span class="o">&lt;=</span> <span class="mi">0</span><span class="p">]</span> <span class="o">=</span> <span class="mi">0</span>
    <span class="k">return</span> <span class="n">dZ</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Ejercicio:</strong> Combinar el cálculo de la derivada de las funciones de activación con la derivada de la parte lineal para obtener, a partir de la derivada de la función de coste respecto la activación de una capa, la derivada de la función de coste respecto a los parámetros de la capa y respecto a las activaciones de la capa anterior.
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[10]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">linear_activation_backward</span><span class="p">(</span><span class="n">dA</span><span class="p">,</span> <span class="n">cache</span><span class="p">,</span> <span class="n">activation</span><span class="p">):</span>
    <span class="sd">&quot;&quot;&quot;</span>
<span class="sd">    Implementa la propagación hacia atrás de una única capa incluyendo la función de activación</span>
<span class="sd">    </span>
<span class="sd">    Argumentos:</span>
<span class="sd">    dA -- derivada de la función de coste con respecto a la salida de la capa actual (l)</span>
<span class="sd">    cache -- dupla que contiene &quot;linear_cache&quot; y &quot;activation_cache&quot;, provinientes de la función linear_activation_forward</span>
<span class="sd">    activation -- el nombre de la función de activación utilizada en la capa actual (l): &quot;sigmoid&quot; o &quot;relu&quot;</span>
<span class="sd">    </span>
<span class="sd">    Devuelve:</span>
<span class="sd">    dA_prev -- derivada de la función de coste con respecto a la salida de la capa anterior (l-1):</span>
<span class="sd">                tiene el mismo tamaño que A_prev</span>
<span class="sd">    dW -- derivada de la función de coste con respecto a la matriz de pesos W de la capa actual (l):</span>
<span class="sd">                tiene el mismo tamaño que W</span>
<span class="sd">    db -- derivada de la función de coste con respecto al vector de bias b de la capa actual (l):</span>
<span class="sd">                tiene el mismo tamaño que b</span>
<span class="sd">    &quot;&quot;&quot;</span>
    
    <span class="n">linear_cache</span><span class="p">,</span> <span class="n">activation_cache</span> <span class="o">=</span> <span class="n">cache</span>
    
    <span class="c1">#print(&quot;linear_activation_backward:Z=&quot;, activation_cache.shape)</span>
    
    <span class="k">if</span> <span class="n">activation</span> <span class="o">==</span> <span class="s2">&quot;relu&quot;</span><span class="p">:</span>
        <span class="n">dZ</span> <span class="o">=</span> <span class="n">relu_backward</span><span class="p">(</span><span class="n">dA</span><span class="p">,</span> <span class="n">activation_cache</span><span class="p">)</span>
    <span class="k">elif</span> <span class="n">activation</span> <span class="o">==</span> <span class="s2">&quot;sigmoid&quot;</span><span class="p">:</span>
        <span class="n">dZ</span> <span class="o">=</span> <span class="n">sigmoid_backward</span><span class="p">(</span><span class="n">dA</span><span class="p">,</span> <span class="n">activation_cache</span><span class="p">)</span>
        
    <span class="n">dA_prev</span><span class="p">,</span> <span class="n">dW</span><span class="p">,</span> <span class="n">db</span> <span class="o">=</span> <span class="n">linear_backward</span><span class="p">(</span><span class="n">dZ</span><span class="p">,</span> <span class="n">linear_cache</span><span class="p">)</span>
    
    <span class="k">return</span> <span class="n">dA_prev</span><span class="p">,</span> <span class="n">dW</span><span class="p">,</span> <span class="n">db</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Por último, es posible calcular la derivada de la función de coste respecto a cualquiera de los parámetros aplicando las funciones recién implementadas empezando por la última capa. Observemos que para inicializar la propagación hacia atrás es necesario calcular primero el valor de $\frac{\partial \mathcal{L}}{\partial A^{[L]}}$.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Ejercicio:</strong> Aplicar la propagación hacia atrás para calcular el gradiente de la función de coste. Observad que el valor de $\frac{\partial \mathcal{L}}{\partial A^{[L]}}$ viene calculado en la variable dAL y que la última capa tiene función de activación sigmoid mientras que todas las demás tienen función de activación relu.
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[11]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">L_model_backward</span><span class="p">(</span><span class="n">AL</span><span class="p">,</span> <span class="n">Y</span><span class="p">,</span> <span class="n">caches</span><span class="p">):</span>
    <span class="sd">&quot;&quot;&quot;</span>
<span class="sd">    Implementa la propagación hacia atrás de la red neuronal completa</span>
<span class="sd">    </span>
<span class="sd">    Argumentos:</span>
<span class="sd">    AL -- salida de la red neuronal, proviene de la función L_model_forward</span>
<span class="sd">    Y -- vector con las etiquetas correctas para cada ejemplo del conjunto de datos: (1, número de ejemplos)</span>
<span class="sd">    caches -- lista de caches que contiene todas las caches de la función linear_activation_forward(), las caches</span>
<span class="sd">                indexadas de 0 a L-2 corresponden a las caches de la función de activación relu y la cache indexada</span>
<span class="sd">                como L-1 corresponde a la cache de la función de activación sigmoid</span>
<span class="sd">    </span>
<span class="sd">    Devuelve:</span>
<span class="sd">    grads -- Un diccionario con las derivadas de la función de coste respecto de cada variable:</span>
<span class="sd">             grads[&quot;dA&quot; + str(l)] = ... </span>
<span class="sd">             grads[&quot;dW&quot; + str(l)] = ...</span>
<span class="sd">             grads[&quot;db&quot; + str(l)] = ... </span>
<span class="sd">    &quot;&quot;&quot;</span>
    
    <span class="n">grads</span> <span class="o">=</span> <span class="p">{}</span>
    <span class="n">L</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">caches</span><span class="p">)</span>
    <span class="n">m</span> <span class="o">=</span> <span class="n">AL</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span>
    <span class="n">Y</span> <span class="o">=</span> <span class="n">Y</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="n">AL</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span>
    
    <span class="c1"># Inicialización de la propagación hacia atrás</span>
    <span class="n">dAL</span> <span class="o">=</span> <span class="o">-</span> <span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">divide</span><span class="p">(</span><span class="n">Y</span><span class="p">,</span> <span class="n">AL</span><span class="p">)</span> <span class="o">-</span> <span class="n">np</span><span class="o">.</span><span class="n">divide</span><span class="p">(</span><span class="mi">1</span> <span class="o">-</span> <span class="n">Y</span><span class="p">,</span> <span class="mi">1</span> <span class="o">-</span> <span class="n">AL</span><span class="p">))</span>
    
    <span class="c1"># Gradiente de la última capa</span>
    <span class="n">current_cache</span> <span class="o">=</span> <span class="n">caches</span><span class="p">[</span><span class="n">L</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span>
    
    <span class="n">activation</span> <span class="o">=</span> <span class="s2">&quot;sigmoid&quot;</span>
    <span class="n">dA_prev_temp</span><span class="p">,</span> <span class="n">dW_temp</span><span class="p">,</span> <span class="n">db_temp</span> <span class="o">=</span> <span class="n">linear_activation_backward</span><span class="p">(</span><span class="n">dAL</span><span class="p">,</span> <span class="n">current_cache</span><span class="p">,</span> <span class="n">activation</span><span class="p">)</span>
    
    <span class="n">grads</span><span class="p">[</span><span class="s2">&quot;dA&quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">L</span><span class="p">)]</span> <span class="o">=</span> <span class="n">dA_prev_temp</span>
    <span class="n">grads</span><span class="p">[</span><span class="s2">&quot;dW&quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">L</span><span class="p">)]</span> <span class="o">=</span> <span class="n">dW_temp</span>
    <span class="n">grads</span><span class="p">[</span><span class="s2">&quot;db&quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">L</span><span class="p">)]</span> <span class="o">=</span> <span class="n">db_temp</span>
    
    <span class="n">activation</span> <span class="o">=</span> <span class="s2">&quot;relu&quot;</span>
    <span class="c1"># Gradiente de las capas restantes</span>
    <span class="k">for</span> <span class="n">l</span> <span class="ow">in</span> <span class="nb">reversed</span><span class="p">(</span><span class="nb">range</span><span class="p">(</span><span class="n">L</span><span class="o">-</span><span class="mi">1</span><span class="p">)):</span>
        <span class="n">current_cache</span> <span class="o">=</span> <span class="n">caches</span><span class="p">[</span><span class="n">l</span><span class="p">]</span> 
        
        <span class="n">dA_curr</span> <span class="o">=</span> <span class="n">dA_prev_temp</span>
        
        <span class="n">dA_prev_temp</span><span class="p">,</span> <span class="n">dW_temp</span><span class="p">,</span> <span class="n">db_temp</span> <span class="o">=</span> <span class="n">linear_activation_backward</span><span class="p">(</span><span class="n">dA_curr</span><span class="p">,</span> <span class="n">current_cache</span><span class="p">,</span> <span class="n">activation</span><span class="p">)</span>
        <span class="n">grads</span><span class="p">[</span><span class="s2">&quot;dA&quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">l</span> <span class="o">+</span> <span class="mi">1</span><span class="p">)]</span> <span class="o">=</span> <span class="n">dA_prev_temp</span>
        <span class="n">grads</span><span class="p">[</span><span class="s2">&quot;dW&quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">l</span> <span class="o">+</span> <span class="mi">1</span><span class="p">)]</span> <span class="o">=</span> <span class="n">dW_temp</span>
        <span class="n">grads</span><span class="p">[</span><span class="s2">&quot;db&quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">l</span> <span class="o">+</span> <span class="mi">1</span><span class="p">)]</span> <span class="o">=</span> <span class="n">db_temp</span>

    <span class="k">return</span> <span class="n">grads</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="1.5-Actualizaci&#243;n-de-par&#225;metros">1.5 Actualizaci&#243;n de par&#225;metros<a class="anchor-link" href="#1.5-Actualizaci&#243;n-de-par&#225;metros">&#182;</a></h3><p>Una vez disponemos del gradiente de la función de coste podemos utilizar el método del descenso del gradiente para actualizar los parámetros de la red neuronal. Si denotamos $\alpha$ la velocidad de aprendizaje, las fórmulas para aplicar un paso del descenso del gradiente son:</p>
$$ W^{[l]} = W^{[l]} - \alpha \text{ } dW^{[l]}$$$$ b^{[l]} = b^{[l]} - \alpha \text{ } db^{[l]}$$
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Ejercicio:</strong> Actualizar los parámetros de la red neuronal aplicando un paso del descenso del gradiente.
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[12]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">update_parameters</span><span class="p">(</span><span class="n">parameters</span><span class="p">,</span> <span class="n">grads</span><span class="p">,</span> <span class="n">learning_rate</span><span class="p">):</span>
    <span class="sd">&quot;&quot;&quot;</span>
<span class="sd">    Actualiza los parámetros utilizando el descenso del gradiente</span>
<span class="sd">    </span>
<span class="sd">    Argumentos:</span>
<span class="sd">    parameters -- diccionario que contiene los parámetros de la red neuronal</span>
<span class="sd">    grads -- diccionario con las derivadas de la función de coste respecto a cada parámetro,</span>
<span class="sd">                corresponde a la salida de la función L_model_backward</span>
<span class="sd">    </span>
<span class="sd">    Devuelve:</span>
<span class="sd">    parameters -- diccionario con los parámetros actualizados:</span>
<span class="sd">                  parameters[&quot;W&quot; + str(l)] = ... </span>
<span class="sd">                  parameters[&quot;b&quot; + str(l)] = ...</span>
<span class="sd">    &quot;&quot;&quot;</span>
    
    <span class="n">L</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">parameters</span><span class="p">)</span> <span class="o">//</span> <span class="mi">2</span>

    <span class="k">for</span> <span class="n">l</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">L</span><span class="p">):</span>
        <span class="n">parameters</span><span class="p">[</span><span class="s2">&quot;W&quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">l</span><span class="o">+</span><span class="mi">1</span><span class="p">)]</span> <span class="o">=</span> <span class="n">parameters</span><span class="p">[</span><span class="s2">&quot;W&quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">l</span><span class="o">+</span><span class="mi">1</span><span class="p">)]</span> <span class="o">-</span> <span class="n">learning_rate</span> <span class="o">*</span> <span class="n">grads</span><span class="p">[</span><span class="s2">&quot;dW&quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">l</span> <span class="o">+</span> <span class="mi">1</span><span class="p">)]</span>
        <span class="n">parameters</span><span class="p">[</span><span class="s2">&quot;b&quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">l</span><span class="o">+</span><span class="mi">1</span><span class="p">)]</span> <span class="o">=</span> <span class="n">parameters</span><span class="p">[</span><span class="s2">&quot;b&quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">l</span><span class="o">+</span><span class="mi">1</span><span class="p">)]</span> <span class="o">-</span> <span class="n">learning_rate</span> <span class="o">*</span> <span class="n">grads</span><span class="p">[</span><span class="s2">&quot;db&quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">l</span> <span class="o">+</span> <span class="mi">1</span><span class="p">)]</span>

    <span class="k">return</span> <span class="n">parameters</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Con todo esto es posible entrenar la red neuronal combinando las funciones definidas anteriormente para aplicar diversas iteraciones del descenso del gradiente e ir actualizando los parámetros de la red de forma reiterada.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>El siguiente código muestra cómo entrenar la red neuronal que hemos construido utilizando únicamente la librería numpy.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[13]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">L_layer_model</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">Y</span><span class="p">,</span> <span class="n">layers_dims</span><span class="p">,</span> <span class="n">learning_rate</span><span class="p">,</span> <span class="n">num_iterations</span><span class="p">,</span> <span class="n">print_cost</span><span class="p">):</span>
    <span class="sd">&quot;&quot;&quot;</span>
<span class="sd">    Implementa una red neuronal de L capas donde las L-1 primeras capas tienen función de activación relu y </span>
<span class="sd">    la última capa tiene función de activación sigmoid.</span>
<span class="sd">    </span>
<span class="sd">    Argumentos:</span>
<span class="sd">    X -- datos: matriz de tamaño (número de variables, número de ejemplos)</span>
<span class="sd">    Y -- vector con las etiquetas correctas para cada ejemplo del conjunto de datos: (1, número de ejemplos)</span>
<span class="sd">    layers_dims -- lista de longitud (número de capas + 1) que contiene el número de variables y el número </span>
<span class="sd">                    de neuronas en cada capa, </span>
<span class="sd">    learning_rate -- velocidad de aprendizaje para aplicar el método del descenso del gradiente</span>
<span class="sd">    num_iterations -- número de pasos para aplicar el descenso del gradiente</span>
<span class="sd">    print_cost -- si el valor es True, escribe el valor de la función de coste cada 10 iteraciones</span>
<span class="sd">    </span>
<span class="sd">    Devuelve:</span>
<span class="sd">    parameters -- parámetros ajustados de la red neuronal</span>
<span class="sd">    &quot;&quot;&quot;</span>
    
    <span class="c1"># Inicialización de los parámetros</span>
    <span class="n">parameters</span> <span class="o">=</span> <span class="n">initialize_parameters</span><span class="p">(</span><span class="n">layers_dims</span><span class="p">)</span>
    
    <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="n">num_iterations</span><span class="p">):</span>
        <span class="c1"># Propagación hacia delante</span>
        <span class="n">AL</span><span class="p">,</span> <span class="n">caches</span> <span class="o">=</span> <span class="n">L_model_forward</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">parameters</span><span class="p">)</span>
        
        <span class="c1"># Cálculo de la función de coste</span>
        <span class="n">cost</span> <span class="o">=</span> <span class="n">compute_cost</span><span class="p">(</span><span class="n">AL</span><span class="p">,</span> <span class="n">Y</span><span class="p">)</span>
    
        <span class="c1"># Propagación hacia atrás</span>
        <span class="n">grads</span> <span class="o">=</span> <span class="n">L_model_backward</span><span class="p">(</span><span class="n">AL</span><span class="p">,</span> <span class="n">Y</span><span class="p">,</span> <span class="n">caches</span><span class="p">)</span>
 
        <span class="c1"># Actualización de parámetros</span>
        <span class="n">parameters</span> <span class="o">=</span> <span class="n">update_parameters</span><span class="p">(</span><span class="n">parameters</span><span class="p">,</span> <span class="n">grads</span><span class="p">,</span> <span class="n">learning_rate</span><span class="p">)</span>
                
        <span class="c1"># Escribe el valor de la función de coste cada 10 iteraciones</span>
        <span class="k">if</span> <span class="n">print_cost</span> <span class="ow">and</span> <span class="n">i</span> <span class="o">%</span> <span class="mi">10</span> <span class="o">==</span> <span class="mi">0</span><span class="p">:</span>
            <span class="nb">print</span> <span class="p">(</span><span class="s2">&quot;Cost after iteration </span><span class="si">%i</span><span class="s2">: </span><span class="si">%f</span><span class="s2">&quot;</span> <span class="o">%</span><span class="p">(</span><span class="n">i</span><span class="p">,</span> <span class="n">cost</span><span class="p">))</span>
    
    <span class="k">return</span> <span class="n">parameters</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="2.-Redes-neuronales-utilizando-Keras">2. Redes neuronales utilizando Keras<a class="anchor-link" href="#2.-Redes-neuronales-utilizando-Keras">&#182;</a></h2><p>A continuación definiremos una red neuronal completamente conectada igual a la que hemos implementado anteriormente pero esta vez utilizando la librería Keras.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Ejercicio:</strong> Definir una red neuronal completamente conectada a partir de una lista que contiene el número de neuronas que debe tener cada capa de la red. Las primeras capas deben tener función de activación relu y la última capa debe tener función de activación sigmoid. Todas ellas tienen que tener kernel_initializer="random_normal" y bias_initializer="zeros".
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[14]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">tensorflow.keras</span> <span class="k">import</span> <span class="n">backend</span>
<span class="kn">from</span> <span class="nn">keras.models</span> <span class="k">import</span> <span class="n">Sequential</span>
<span class="kn">from</span> <span class="nn">keras.layers</span> <span class="k">import</span> <span class="n">Dense</span>
<span class="kn">from</span> <span class="nn">keras.optimizers</span> <span class="k">import</span> <span class="n">SGD</span>

<span class="k">def</span> <span class="nf">keras_model</span><span class="p">(</span><span class="n">layers_dims</span><span class="p">,</span> <span class="n">learning_rate</span><span class="p">):</span>
    <span class="sd">&quot;&quot;&quot;</span>
<span class="sd">    Crea, utilizando Keras, una red neuronal de L capas completamente conectadas donde las L-1 primeras capas</span>
<span class="sd">    tienen función de activación relu y la última capa tiene función de activación sigmoid.</span>
<span class="sd">    </span>
<span class="sd">    Argumentos:</span>
<span class="sd">    layers_dims -- lista de longitud (número de capas + 1) que contiene el número de variables y el número </span>
<span class="sd">                    de neuronas en cada capa, </span>
<span class="sd">    learning_rate -- velocidad de aprendizaje para aplicar el método del descenso del gradiente</span>
<span class="sd">    </span>
<span class="sd">    Devuelve:</span>
<span class="sd">    modelo -- objeto de Keras que representa la red neuronal</span>
<span class="sd">    &quot;&quot;&quot;</span>
    
    <span class="n">L</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">layers_dims</span><span class="p">)</span>
    
    <span class="n">model</span> <span class="o">=</span> <span class="n">Sequential</span><span class="p">()</span>
    
    <span class="c1"># Añadir L-1 capas con función de activación relu y una última capa con función de activación sigmoid,</span>
    <span class="c1"># cada capa debe tener el número de neuronas indicado en la variable layers_dims, el tamaño de la capa</span>
    <span class="c1"># de entrada viene dado en layers_dims[0]</span>
    <span class="k">for</span> <span class="n">l</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">L</span><span class="o">-</span><span class="mi">2</span><span class="p">):</span>
        <span class="n">model</span><span class="o">.</span><span class="n">add</span><span class="p">(</span><span class="n">Dense</span><span class="p">(</span><span class="n">layers_dims</span><span class="p">[</span><span class="n">l</span><span class="p">],</span> <span class="n">activation</span> <span class="o">=</span> <span class="s1">&#39;relu&#39;</span><span class="p">))</span>
        
    <span class="n">model</span><span class="o">.</span><span class="n">add</span><span class="p">(</span><span class="n">Dense</span><span class="p">(</span><span class="n">layers_dims</span><span class="p">[</span><span class="n">L</span><span class="o">-</span><span class="mi">1</span><span class="p">],</span> <span class="n">activation</span> <span class="o">=</span> <span class="s1">&#39;sigmoid&#39;</span><span class="p">))</span>
    
    
    <span class="n">model</span><span class="o">.</span><span class="n">compile</span><span class="p">(</span><span class="n">optimizer</span><span class="o">=</span><span class="n">SGD</span><span class="p">(</span><span class="n">lr</span><span class="o">=</span><span class="n">learning_rate</span><span class="p">),</span> <span class="n">loss</span><span class="o">=</span><span class="s1">&#39;binary_crossentropy&#39;</span><span class="p">,</span> <span class="n">metrics</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;accuracy&#39;</span><span class="p">])</span>
    
    <span class="k">return</span> <span class="n">model</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stderr output_text">
<pre>Using TensorFlow backend.
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="3.-Entrenamiento-de-la-red-neuronal">3. Entrenamiento de la red neuronal<a class="anchor-link" href="#3.-Entrenamiento-de-la-red-neuronal">&#182;</a></h2><p>Con todas las funciones implementadas anteriormente es posible entrenar una red neuronal completamente conectada con cualquier número de capas y cualquier número de neuronas en cada capa.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>A continuación definimos la estructura de capas que tendrá la red neuronal.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[15]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">layers_dims</span> <span class="o">=</span> <span class="p">[</span><span class="mi">100</span><span class="p">,</span> <span class="mi">80</span><span class="p">,</span> <span class="mi">50</span><span class="p">,</span> <span class="mi">40</span><span class="p">,</span> <span class="mi">20</span><span class="p">,</span> <span class="mi">5</span><span class="p">,</span> <span class="mi">1</span><span class="p">]</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Para entrenar la red neuronal que hemos construido únicamente utilizando numpy debemos ejecutar el siguiente código:</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[16]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">parameters</span> <span class="o">=</span> <span class="n">L_layer_model</span><span class="p">(</span><span class="n">train_x</span><span class="o">.</span><span class="n">T</span><span class="p">,</span> <span class="n">train_y</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="o">-</span><span class="mi">1</span><span class="p">),</span> <span class="n">layers_dims</span><span class="o">=</span><span class="n">layers_dims</span><span class="p">,</span> <span class="n">learning_rate</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span> 
                           <span class="n">num_iterations</span><span class="o">=</span><span class="mi">250</span><span class="p">,</span> <span class="n">print_cost</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Cost after iteration 0: 0.695712
Cost after iteration 10: 0.694522
Cost after iteration 20: 0.693761
Cost after iteration 30: 0.693244
Cost after iteration 40: 0.692860
Cost after iteration 50: 0.692537
Cost after iteration 60: 0.692231
Cost after iteration 70: 0.691904
Cost after iteration 80: 0.691525
Cost after iteration 90: 0.691052
Cost after iteration 100: 0.690431
Cost after iteration 110: 0.689565
Cost after iteration 120: 0.688282
Cost after iteration 130: 0.686319
Cost after iteration 140: 0.683359
Cost after iteration 150: 0.678627
Cost after iteration 160: 0.670799
Cost after iteration 170: 0.657658
Cost after iteration 180: 0.633955
Cost after iteration 190: 0.587569
Cost after iteration 200: 0.497403
Cost after iteration 210: 0.357097
Cost after iteration 220: 0.212702
Cost after iteration 230: 0.124110
Cost after iteration 240: 0.087445
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Para entrenar la red neuronal que hemos construido utilizando Keras debemos ejecutar el siguiente código:</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[17]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model</span> <span class="o">=</span> <span class="n">keras_model</span><span class="p">(</span><span class="n">layers_dims</span> <span class="o">=</span> <span class="n">layers_dims</span><span class="p">,</span> <span class="n">learning_rate</span> <span class="o">=</span> <span class="mf">0.1</span><span class="p">)</span>
<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">train_x</span><span class="p">,</span> <span class="n">train_y</span><span class="p">,</span> <span class="n">epochs</span><span class="o">=</span><span class="mi">250</span><span class="p">,</span> <span class="n">batch_size</span><span class="o">=</span><span class="n">train_x</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">verbose</span><span class="o">=</span><span class="mi">2</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Epoch 1/250
 - 1s - loss: 0.7099 - acc: 0.4350
Epoch 2/250
 - 0s - loss: 0.6738 - acc: 0.7056
Epoch 3/250
 - 0s - loss: 0.6453 - acc: 0.8062
Epoch 4/250
 - 0s - loss: 0.6174 - acc: 0.8456
Epoch 5/250
 - 0s - loss: 0.5870 - acc: 0.8681
Epoch 6/250
 - 0s - loss: 0.5522 - acc: 0.8788
Epoch 7/250
 - 0s - loss: 0.5153 - acc: 0.8894
Epoch 8/250
 - 0s - loss: 0.4763 - acc: 0.8988
Epoch 9/250
 - 0s - loss: 0.4360 - acc: 0.9044
Epoch 10/250
 - 0s - loss: 0.3965 - acc: 0.9137
Epoch 11/250
 - 0s - loss: 0.3584 - acc: 0.9200
Epoch 12/250
 - 0s - loss: 0.3236 - acc: 0.9250
Epoch 13/250
 - 0s - loss: 0.2926 - acc: 0.9294
Epoch 14/250
 - 0s - loss: 0.2658 - acc: 0.9325
Epoch 15/250
 - 0s - loss: 0.2427 - acc: 0.9350
Epoch 16/250
 - 0s - loss: 0.2229 - acc: 0.9356
Epoch 17/250
 - 0s - loss: 0.2060 - acc: 0.9375
Epoch 18/250
 - 0s - loss: 0.1917 - acc: 0.9400
Epoch 19/250
 - 0s - loss: 0.1795 - acc: 0.9419
Epoch 20/250
 - 0s - loss: 0.1690 - acc: 0.9425
Epoch 21/250
 - 0s - loss: 0.1598 - acc: 0.9438
Epoch 22/250
 - 0s - loss: 0.1517 - acc: 0.9450
Epoch 23/250
 - 0s - loss: 0.1446 - acc: 0.9463
Epoch 24/250
 - 0s - loss: 0.1382 - acc: 0.9513
Epoch 25/250
 - 0s - loss: 0.1324 - acc: 0.9525
Epoch 26/250
 - 0s - loss: 0.1271 - acc: 0.9538
Epoch 27/250
 - 0s - loss: 0.1222 - acc: 0.9550
Epoch 28/250
 - 0s - loss: 0.1176 - acc: 0.9575
Epoch 29/250
 - 0s - loss: 0.1134 - acc: 0.9575
Epoch 30/250
 - 0s - loss: 0.1094 - acc: 0.9569
Epoch 31/250
 - 0s - loss: 0.1057 - acc: 0.9581
Epoch 32/250
 - 0s - loss: 0.1022 - acc: 0.9594
Epoch 33/250
 - 0s - loss: 0.0989 - acc: 0.9613
Epoch 34/250
 - 0s - loss: 0.0958 - acc: 0.9619
Epoch 35/250
 - 0s - loss: 0.0929 - acc: 0.9631
Epoch 36/250
 - 0s - loss: 0.0901 - acc: 0.9631
Epoch 37/250
 - 0s - loss: 0.0875 - acc: 0.9638
Epoch 38/250
 - 0s - loss: 0.0849 - acc: 0.9669
Epoch 39/250
 - 0s - loss: 0.0826 - acc: 0.9675
Epoch 40/250
 - 0s - loss: 0.0803 - acc: 0.9681
Epoch 41/250
 - 0s - loss: 0.0782 - acc: 0.9694
Epoch 42/250
 - 0s - loss: 0.0761 - acc: 0.9719
Epoch 43/250
 - 0s - loss: 0.0742 - acc: 0.9725
Epoch 44/250
 - 0s - loss: 0.0724 - acc: 0.9731
Epoch 45/250
 - 0s - loss: 0.0707 - acc: 0.9737
Epoch 46/250
 - 0s - loss: 0.0691 - acc: 0.9737
Epoch 47/250
 - 0s - loss: 0.0676 - acc: 0.9756
Epoch 48/250
 - 0s - loss: 0.0661 - acc: 0.9762
Epoch 49/250
 - 0s - loss: 0.0647 - acc: 0.9775
Epoch 50/250
 - 0s - loss: 0.0634 - acc: 0.9775
Epoch 51/250
 - 0s - loss: 0.0621 - acc: 0.9775
Epoch 52/250
 - 0s - loss: 0.0609 - acc: 0.9781
Epoch 53/250
 - 0s - loss: 0.0598 - acc: 0.9787
Epoch 54/250
 - 0s - loss: 0.0587 - acc: 0.9800
Epoch 55/250
 - 0s - loss: 0.0577 - acc: 0.9800
Epoch 56/250
 - 0s - loss: 0.0567 - acc: 0.9806
Epoch 57/250
 - 0s - loss: 0.0557 - acc: 0.9825
Epoch 58/250
 - 0s - loss: 0.0548 - acc: 0.9825
Epoch 59/250
 - 0s - loss: 0.0539 - acc: 0.9825
Epoch 60/250
 - 0s - loss: 0.0530 - acc: 0.9825
Epoch 61/250
 - 0s - loss: 0.0522 - acc: 0.9825
Epoch 62/250
 - 0s - loss: 0.0513 - acc: 0.9825
Epoch 63/250
 - 0s - loss: 0.0506 - acc: 0.9825
Epoch 64/250
 - 0s - loss: 0.0498 - acc: 0.9825
Epoch 65/250
 - 0s - loss: 0.0491 - acc: 0.9825
Epoch 66/250
 - 0s - loss: 0.0484 - acc: 0.9825
Epoch 67/250
 - 0s - loss: 0.0477 - acc: 0.9825
Epoch 68/250
 - 0s - loss: 0.0470 - acc: 0.9825
Epoch 69/250
 - 0s - loss: 0.0464 - acc: 0.9819
Epoch 70/250
 - 0s - loss: 0.0457 - acc: 0.9831
Epoch 71/250
 - 0s - loss: 0.0451 - acc: 0.9831
Epoch 72/250
 - 0s - loss: 0.0445 - acc: 0.9831
Epoch 73/250
 - 0s - loss: 0.0439 - acc: 0.9831
Epoch 74/250
 - 0s - loss: 0.0433 - acc: 0.9831
Epoch 75/250
 - 0s - loss: 0.0427 - acc: 0.9831
Epoch 76/250
 - 0s - loss: 0.0422 - acc: 0.9837
Epoch 77/250
 - 0s - loss: 0.0416 - acc: 0.9837
Epoch 78/250
 - 0s - loss: 0.0411 - acc: 0.9844
Epoch 79/250
 - 0s - loss: 0.0406 - acc: 0.9844
Epoch 80/250
 - 0s - loss: 0.0401 - acc: 0.9850
Epoch 81/250
 - 0s - loss: 0.0396 - acc: 0.9850
Epoch 82/250
 - 0s - loss: 0.0391 - acc: 0.9850
Epoch 83/250
 - 0s - loss: 0.0386 - acc: 0.9856
Epoch 84/250
 - 0s - loss: 0.0382 - acc: 0.9862
Epoch 85/250
 - 0s - loss: 0.0377 - acc: 0.9862
Epoch 86/250
 - 0s - loss: 0.0373 - acc: 0.9869
Epoch 87/250
 - 0s - loss: 0.0369 - acc: 0.9875
Epoch 88/250
 - 0s - loss: 0.0364 - acc: 0.9875
Epoch 89/250
 - 0s - loss: 0.0360 - acc: 0.9881
Epoch 90/250
 - 0s - loss: 0.0356 - acc: 0.9881
Epoch 91/250
 - 0s - loss: 0.0352 - acc: 0.9881
Epoch 92/250
 - 0s - loss: 0.0348 - acc: 0.9887
Epoch 93/250
 - 0s - loss: 0.0344 - acc: 0.9887
Epoch 94/250
 - 0s - loss: 0.0341 - acc: 0.9894
Epoch 95/250
 - 0s - loss: 0.0337 - acc: 0.9894
Epoch 96/250
 - 0s - loss: 0.0333 - acc: 0.9894
Epoch 97/250
 - 0s - loss: 0.0330 - acc: 0.9894
Epoch 98/250
 - 0s - loss: 0.0326 - acc: 0.9894
Epoch 99/250
 - 0s - loss: 0.0323 - acc: 0.9894
Epoch 100/250
 - 0s - loss: 0.0319 - acc: 0.9894
Epoch 101/250
 - 0s - loss: 0.0316 - acc: 0.9894
Epoch 102/250
 - 0s - loss: 0.0313 - acc: 0.9894
Epoch 103/250
 - 0s - loss: 0.0309 - acc: 0.9894
Epoch 104/250
 - 0s - loss: 0.0306 - acc: 0.9894
Epoch 105/250
 - 0s - loss: 0.0303 - acc: 0.9894
Epoch 106/250
 - 0s - loss: 0.0299 - acc: 0.9894
Epoch 107/250
 - 0s - loss: 0.0296 - acc: 0.9894
Epoch 108/250
 - 0s - loss: 0.0293 - acc: 0.9894
Epoch 109/250
 - 0s - loss: 0.0290 - acc: 0.9894
Epoch 110/250
 - 0s - loss: 0.0287 - acc: 0.9900
Epoch 111/250
 - 0s - loss: 0.0284 - acc: 0.9900
Epoch 112/250
 - 0s - loss: 0.0281 - acc: 0.9900
Epoch 113/250
 - 0s - loss: 0.0278 - acc: 0.9900
Epoch 114/250
 - 0s - loss: 0.0275 - acc: 0.9906
Epoch 115/250
 - 0s - loss: 0.0272 - acc: 0.9906
Epoch 116/250
 - 0s - loss: 0.0269 - acc: 0.9906
Epoch 117/250
 - 0s - loss: 0.0266 - acc: 0.9906
Epoch 118/250
 - 0s - loss: 0.0263 - acc: 0.9906
Epoch 119/250
 - 0s - loss: 0.0260 - acc: 0.9906
Epoch 120/250
 - 0s - loss: 0.0258 - acc: 0.9906
Epoch 121/250
 - 0s - loss: 0.0255 - acc: 0.9906
Epoch 122/250
 - 0s - loss: 0.0252 - acc: 0.9912
Epoch 123/250
 - 0s - loss: 0.0250 - acc: 0.9912
Epoch 124/250
 - 0s - loss: 0.0247 - acc: 0.9912
Epoch 125/250
 - 0s - loss: 0.0244 - acc: 0.9912
Epoch 126/250
 - 0s - loss: 0.0242 - acc: 0.9912
Epoch 127/250
 - 0s - loss: 0.0239 - acc: 0.9912
Epoch 128/250
 - 0s - loss: 0.0237 - acc: 0.9912
Epoch 129/250
 - 0s - loss: 0.0234 - acc: 0.9912
Epoch 130/250
 - 0s - loss: 0.0231 - acc: 0.9919
Epoch 131/250
 - 0s - loss: 0.0229 - acc: 0.9919
Epoch 132/250
 - 0s - loss: 0.0227 - acc: 0.9919
Epoch 133/250
 - 0s - loss: 0.0224 - acc: 0.9925
Epoch 134/250
 - 0s - loss: 0.0222 - acc: 0.9925
Epoch 135/250
 - 0s - loss: 0.0219 - acc: 0.9925
Epoch 136/250
 - 0s - loss: 0.0217 - acc: 0.9925
Epoch 137/250
 - 0s - loss: 0.0215 - acc: 0.9925
Epoch 138/250
 - 0s - loss: 0.0212 - acc: 0.9925
Epoch 139/250
 - 0s - loss: 0.0210 - acc: 0.9925
Epoch 140/250
 - 0s - loss: 0.0208 - acc: 0.9925
Epoch 141/250
 - 0s - loss: 0.0206 - acc: 0.9925
Epoch 142/250
 - 0s - loss: 0.0204 - acc: 0.9925
Epoch 143/250
 - 0s - loss: 0.0201 - acc: 0.9925
Epoch 144/250
 - 0s - loss: 0.0199 - acc: 0.9925
Epoch 145/250
 - 0s - loss: 0.0197 - acc: 0.9925
Epoch 146/250
 - 0s - loss: 0.0195 - acc: 0.9925
Epoch 147/250
 - 0s - loss: 0.0193 - acc: 0.9925
Epoch 148/250
 - 0s - loss: 0.0191 - acc: 0.9925
Epoch 149/250
 - 0s - loss: 0.0189 - acc: 0.9925
Epoch 150/250
 - 0s - loss: 0.0187 - acc: 0.9925
Epoch 151/250
 - 0s - loss: 0.0185 - acc: 0.9925
Epoch 152/250
 - 0s - loss: 0.0183 - acc: 0.9931
Epoch 153/250
 - 0s - loss: 0.0181 - acc: 0.9944
Epoch 154/250
 - 0s - loss: 0.0179 - acc: 0.9944
Epoch 155/250
 - 0s - loss: 0.0177 - acc: 0.9944
Epoch 156/250
 - 0s - loss: 0.0176 - acc: 0.9944
Epoch 157/250
 - 0s - loss: 0.0174 - acc: 0.9944
Epoch 158/250
 - 0s - loss: 0.0172 - acc: 0.9944
Epoch 159/250
 - 0s - loss: 0.0170 - acc: 0.9944
Epoch 160/250
 - 0s - loss: 0.0168 - acc: 0.9944
Epoch 161/250
 - 0s - loss: 0.0167 - acc: 0.9950
Epoch 162/250
 - 0s - loss: 0.0165 - acc: 0.9950
Epoch 163/250
 - 0s - loss: 0.0163 - acc: 0.9950
Epoch 164/250
 - 0s - loss: 0.0162 - acc: 0.9956
Epoch 165/250
 - 0s - loss: 0.0160 - acc: 0.9956
Epoch 166/250
 - 0s - loss: 0.0158 - acc: 0.9956
Epoch 167/250
 - 0s - loss: 0.0157 - acc: 0.9956
Epoch 168/250
 - 0s - loss: 0.0155 - acc: 0.9956
Epoch 169/250
 - 0s - loss: 0.0153 - acc: 0.9956
Epoch 170/250
 - 0s - loss: 0.0152 - acc: 0.9956
Epoch 171/250
 - 0s - loss: 0.0150 - acc: 0.9956
Epoch 172/250
 - 0s - loss: 0.0149 - acc: 0.9956
Epoch 173/250
 - 0s - loss: 0.0147 - acc: 0.9956
Epoch 174/250
 - 0s - loss: 0.0145 - acc: 0.9956
Epoch 175/250
 - 0s - loss: 0.0144 - acc: 0.9956
Epoch 176/250
 - 0s - loss: 0.0142 - acc: 0.9956
Epoch 177/250
 - 0s - loss: 0.0141 - acc: 0.9956
Epoch 178/250
 - 0s - loss: 0.0139 - acc: 0.9956
Epoch 179/250
 - 0s - loss: 0.0138 - acc: 0.9956
Epoch 180/250
 - 0s - loss: 0.0136 - acc: 0.9956
Epoch 181/250
 - 0s - loss: 0.0135 - acc: 0.9956
Epoch 182/250
 - 0s - loss: 0.0134 - acc: 0.9956
Epoch 183/250
 - 0s - loss: 0.0132 - acc: 0.9962
Epoch 184/250
 - 0s - loss: 0.0131 - acc: 0.9962
Epoch 185/250
 - 0s - loss: 0.0129 - acc: 0.9962
Epoch 186/250
 - 0s - loss: 0.0128 - acc: 0.9962
Epoch 187/250
 - 0s - loss: 0.0127 - acc: 0.9962
Epoch 188/250
 - 0s - loss: 0.0125 - acc: 0.9962
Epoch 189/250
 - 0s - loss: 0.0124 - acc: 0.9962
Epoch 190/250
 - 0s - loss: 0.0123 - acc: 0.9962
Epoch 191/250
 - 0s - loss: 0.0121 - acc: 0.9962
Epoch 192/250
 - 0s - loss: 0.0120 - acc: 0.9962
Epoch 193/250
 - 0s - loss: 0.0119 - acc: 0.9962
Epoch 194/250
 - 0s - loss: 0.0118 - acc: 0.9962
Epoch 195/250
 - 0s - loss: 0.0116 - acc: 0.9962
Epoch 196/250
 - 0s - loss: 0.0115 - acc: 0.9962
Epoch 197/250
 - 0s - loss: 0.0114 - acc: 0.9962
Epoch 198/250
 - 0s - loss: 0.0113 - acc: 0.9962
Epoch 199/250
 - 0s - loss: 0.0112 - acc: 0.9962
Epoch 200/250
 - 0s - loss: 0.0110 - acc: 0.9962
Epoch 201/250
 - 0s - loss: 0.0109 - acc: 0.9962
Epoch 202/250
 - 0s - loss: 0.0108 - acc: 0.9962
Epoch 203/250
 - 0s - loss: 0.0107 - acc: 0.9962
Epoch 204/250
 - 0s - loss: 0.0106 - acc: 0.9975
Epoch 205/250
 - 0s - loss: 0.0105 - acc: 0.9975
Epoch 206/250
 - 0s - loss: 0.0104 - acc: 0.9975
Epoch 207/250
 - 0s - loss: 0.0103 - acc: 0.9975
Epoch 208/250
 - 0s - loss: 0.0102 - acc: 0.9975
Epoch 209/250
 - 0s - loss: 0.0100 - acc: 0.9981
Epoch 210/250
 - 0s - loss: 0.0099 - acc: 0.9987
Epoch 211/250
 - 0s - loss: 0.0098 - acc: 0.9987
Epoch 212/250
 - 0s - loss: 0.0097 - acc: 0.9987
Epoch 213/250
 - 0s - loss: 0.0096 - acc: 0.9987
Epoch 214/250
 - 0s - loss: 0.0095 - acc: 0.9987
Epoch 215/250
 - 0s - loss: 0.0094 - acc: 0.9987
Epoch 216/250
 - 0s - loss: 0.0093 - acc: 0.9987
Epoch 217/250
 - 0s - loss: 0.0092 - acc: 0.9987
Epoch 218/250
 - 0s - loss: 0.0091 - acc: 0.9987
Epoch 219/250
 - 0s - loss: 0.0091 - acc: 0.9987
Epoch 220/250
 - 0s - loss: 0.0090 - acc: 0.9987
Epoch 221/250
 - 0s - loss: 0.0089 - acc: 0.9994
Epoch 222/250
 - 0s - loss: 0.0088 - acc: 0.9994
Epoch 223/250
 - 0s - loss: 0.0087 - acc: 0.9994
Epoch 224/250
 - 0s - loss: 0.0086 - acc: 0.9994
Epoch 225/250
 - 0s - loss: 0.0085 - acc: 0.9994
Epoch 226/250
 - 0s - loss: 0.0084 - acc: 0.9994
Epoch 227/250
 - 0s - loss: 0.0083 - acc: 0.9994
Epoch 228/250
 - 0s - loss: 0.0082 - acc: 0.9994
Epoch 229/250
 - 0s - loss: 0.0082 - acc: 0.9994
Epoch 230/250
 - 0s - loss: 0.0081 - acc: 0.9994
Epoch 231/250
 - 0s - loss: 0.0080 - acc: 0.9994
Epoch 232/250
 - 0s - loss: 0.0079 - acc: 0.9994
Epoch 233/250
 - 0s - loss: 0.0078 - acc: 0.9994
Epoch 234/250
 - 0s - loss: 0.0077 - acc: 1.0000
Epoch 235/250
 - 0s - loss: 0.0077 - acc: 1.0000
Epoch 236/250
 - 0s - loss: 0.0076 - acc: 1.0000
Epoch 237/250
 - 0s - loss: 0.0075 - acc: 1.0000
Epoch 238/250
 - 0s - loss: 0.0074 - acc: 1.0000
Epoch 239/250
 - 0s - loss: 0.0074 - acc: 1.0000
Epoch 240/250
 - 0s - loss: 0.0073 - acc: 1.0000
Epoch 241/250
 - 0s - loss: 0.0072 - acc: 1.0000
Epoch 242/250
 - 0s - loss: 0.0071 - acc: 1.0000
Epoch 243/250
 - 0s - loss: 0.0071 - acc: 1.0000
Epoch 244/250
 - 0s - loss: 0.0070 - acc: 1.0000
Epoch 245/250
 - 0s - loss: 0.0069 - acc: 1.0000
Epoch 246/250
 - 0s - loss: 0.0069 - acc: 1.0000
Epoch 247/250
 - 0s - loss: 0.0068 - acc: 1.0000
Epoch 248/250
 - 0s - loss: 0.0067 - acc: 1.0000
Epoch 249/250
 - 0s - loss: 0.0067 - acc: 1.0000
Epoch 250/250
 - 0s - loss: 0.0066 - acc: 1.0000
</pre>
</div>
</div>

<div class="output_area">

    <div class="prompt output_prompt">Out[17]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;keras.callbacks.History at 0xd2264a8&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Por último, podemos utilizar la siguiente función para calcular la precisión que obtenemos con la red neuronal construida utilizando únicamente numpy.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[18]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">accuracy</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">parameters</span><span class="p">):</span>
    <span class="sd">&quot;&quot;&quot;</span>
<span class="sd">    Calcula la precisión de las predicciones de la red neuronal.</span>
<span class="sd">    </span>
<span class="sd">    Argumentos:</span>
<span class="sd">    X -- datos: matriz de tamaño (número de variables, número de ejemplos)</span>
<span class="sd">    parameters -- parámetros de la red neuronal entrenada</span>
<span class="sd">    </span>
<span class="sd">    Returns:</span>
<span class="sd">    accuracy -- valor entre 0 y 1 que representa la precisión de la red neuronal</span>
<span class="sd">    &quot;&quot;&quot;</span>
    
    <span class="n">m</span> <span class="o">=</span> <span class="n">X</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span>
    <span class="n">p</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">zeros</span><span class="p">((</span><span class="mi">1</span><span class="p">,</span><span class="n">m</span><span class="p">))</span>
    
    <span class="c1"># Propagación hacia delante</span>
    <span class="n">probas</span><span class="p">,</span> <span class="n">caches</span> <span class="o">=</span> <span class="n">L_model_forward</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">parameters</span><span class="p">)</span>

    <span class="c1"># Conversión de la salida de la red a valores 0 o 1</span>
    <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="n">probas</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">1</span><span class="p">]):</span>
        <span class="k">if</span> <span class="n">probas</span><span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="n">i</span><span class="p">]</span> <span class="o">&gt;</span> <span class="mf">0.5</span><span class="p">:</span>
            <span class="n">p</span><span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="n">i</span><span class="p">]</span> <span class="o">=</span> <span class="mi">1</span>
        <span class="k">else</span><span class="p">:</span>
            <span class="n">p</span><span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="n">i</span><span class="p">]</span> <span class="o">=</span> <span class="mi">0</span>
            
    <span class="n">accuracy</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">sum</span><span class="p">((</span><span class="n">p</span> <span class="o">==</span> <span class="n">y</span><span class="p">))</span> <span class="o">/</span> <span class="n">m</span>
    
    <span class="k">return</span> <span class="n">accuracy</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>A continuación se muestra la precisión obtenida tanto con la red construida con numpy como con la red construida con Keras.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[19]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Red construida con numpy&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Precisión </span><span class="si">{:.2f}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">accuracy</span><span class="p">(</span><span class="n">test_x</span><span class="o">.</span><span class="n">T</span><span class="p">,</span> <span class="n">test_y</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="o">-</span><span class="mi">1</span><span class="p">),</span> <span class="n">parameters</span><span class="p">)))</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;---&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Red construida con Keras&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Precisión </span><span class="si">{:.2f}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">evaluate</span><span class="p">(</span><span class="n">test_x</span><span class="p">,</span> <span class="n">test_y</span><span class="p">,</span> <span class="n">verbose</span><span class="o">=</span><span class="mi">0</span><span class="p">)[</span><span class="mi">1</span><span class="p">]))</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Red construida con numpy
Precisión 0.96
---
Red construida con Keras
Precisión 0.98
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>El siguiente código permite calcular el tiempo que tarda cada red neuronal en entrenarse.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[20]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="o">%%time</span>it
<span class="n">parameters</span> <span class="o">=</span> <span class="n">L_layer_model</span><span class="p">(</span><span class="n">train_x</span><span class="o">.</span><span class="n">T</span><span class="p">,</span> <span class="n">train_y</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="o">-</span><span class="mi">1</span><span class="p">),</span> <span class="n">layers_dims</span><span class="o">=</span><span class="n">layers_dims</span><span class="p">,</span> 
                           <span class="n">learning_rate</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span> <span class="n">num_iterations</span><span class="o">=</span><span class="mi">250</span><span class="p">,</span> <span class="n">print_cost</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>3.72 s ± 307 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[21]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="o">%%time</span>it
<span class="n">model</span> <span class="o">=</span> <span class="n">keras_model</span><span class="p">(</span><span class="n">layers_dims</span><span class="o">=</span><span class="n">layers_dims</span><span class="p">,</span> <span class="n">learning_rate</span><span class="o">=</span><span class="mf">0.1</span><span class="p">)</span>
<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">train_x</span><span class="p">,</span> <span class="n">train_y</span><span class="p">,</span> <span class="n">epochs</span><span class="o">=</span><span class="mi">250</span><span class="p">,</span> <span class="n">batch_size</span><span class="o">=</span><span class="n">train_x</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">verbose</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>4.09 s ± 239 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Análisis:</strong> Comparar el rendimiento, tanto en tiempo de ejecución como en precisión, de las dos implementaciones de la red neuronal. Utilizar diferentes hiperparámetros en la comparación: probar con diferentes valores para las dimensiones de las capas, diferente número de capas, número de iteraciones, etc. ¿Qué factores pueden estar creando las diferencias observadas?
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<div style="background-color: #C7FCE5; border-color:#9DFCD2  ; border-left: 5px solid #9DFCD2  ; padding: 0.5em;">
    <p>A continuación vamos a visualizar un conjunto de tablas las cuales reflejamos la modificación de los diferentes parámetros como pueden ser <b>el número de iteraciones y la tasa de aprendizaje</b> y visualizamos los resultados en función de <b> la precisión y el tiempo de entrenamiento</b>.</p>
 </div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><b>layers_dims = [100, 20, 5, 1]</b></p>
<p><b>Numpy</b></p>
<table>

<tr>
  <td><strong>learning_rate</strong></td>
  <td><strong>num_iterations</strong></td>
  <td><strong>accuracy</strong></td>
  <td><strong>time(s)</strong></td>
</tr>

<tr>
  <td>0.1</td>
  <td>50</td>
  <td>0.87</td>
  <td>0.152</td>
</tr>

<tr>
  <td>0.5</td>
  <td>50</td>
  <td>0.98</td>
  <td>0.153</td>
</tr>

<tr>
  <td>0.1</td>
  <td>250</td>
  <td>0.98</td>
  <td>0.742</td>
</tr>

<tr>
  <td>0.5</td>
  <td>250</td>
  <td>0.99</td>
  <td>0.750</td>
</tr>

<tr>
  <td>0.1</td>
  <td>500</td>
  <td>0.99</td>
  <td>1.54</td>
</tr>

<tr>
  <td>0.5</td>
  <td>500</td>
  <td>0.99</td>
  <td>1.5</td>
</tr>


</table><p><b>Keras</b></p>
<table>

<tr>
  <td><strong>learning_rate</strong></td>
  <td><strong>num_iterations</strong></td>
  <td><strong>accuracy</strong></td>
  <td><strong>time(s)</strong></td>
</tr>

<tr>
  <td>0.1</td>
  <td>50</td>
  <td>0.97</td>
  <td>3.69</td>
</tr>

<tr>
  <td>0.5</td>
  <td>50</td>
  <td>0.99</td>
  <td>3.54</td>
</tr>


<tr>
  <td>0.1</td>
  <td>250</td>
  <td>0.99</td>
  <td>2.35</td>
</tr>

<tr>
  <td>0.5</td>
  <td>250</td>
  <td>0.99</td>
  <td>4.68</td>
</tr>

<tr>
  <td>0.1</td>
  <td>500</td>
  <td>0.99</td>
  <td>6.12</td>
</tr>

<tr>
  <td>0.5</td>
  <td>500</td>
  <td>0.99</td>
  <td>5.5</td>
</tr>

</table><p><b>layers_dims = [100, 80, 50, 40, 20, 5, 1]</b></p>
<p><b>Numpy</b></p>
<table>

<tr>
  <td><strong>learning_rate</strong></td>
  <td><strong>num_iterations</strong></td>
  <td><strong>accuracy</strong></td>
  <td><strong>time(s)</strong></td>
</tr>

<tr>
  <td>0.1</td>
  <td>50</td>  
  <td>0.48</td>
  <td>4.69</td>
</tr>

<tr>
  <td>0.5</td>
  <td>50</td>
  <td>0.80</td>
  <td>3.57</td>
</tr>

<tr>
  <td>0.1</td>
  <td>250</td>
  <td>0.98</td>
  <td>4.69</td>
</tr>

<tr>
  <td>0.5</td>
  <td>250</td>
  <td>0.98</td>
  <td>3.06</td>
</tr>

<tr>
  <td>10</td>
  <td>250</td>
  <td>0.48</td>
  <td>3.31</td>
</tr>

<tr>
  <td>0.1</td>
  <td>500</td>
  <td>0.99</td>
  <td>3.54</td>
</tr>

<tr>
  <td>0.5</td>
  <td>500</td>
  <td>0.99</td>
  <td>3.41</td>
</tr>


</table><p><b>Keras</b></p>
<table>

<tr>
  <td><strong>learning_rate</strong></td>
  <td><strong>num_iterations</strong></td>
  <td><strong>accuracy</strong></td>
  <td><strong>time(s)</strong></td>
</tr>

<tr>
  <td>0.1</td>
  <td>50</td>  
  <td>0.96</td>
  <td>6.55</td>
</tr>

<tr>
  <td>0.5</td>
  <td>50</td>  
  <td>0.52</td>
  <td>7.8</td>
</tr>


<tr>
  <td>0.1</td>
  <td>250</td>
  <td>0.99</td>
  <td>6.55</td>
</tr>

<tr>
  <td>0.5</td>
  <td>250</td>
  <td>0.99</td>
  <td>7.06</td>
</tr>

<tr>
  <td>10</td>
  <td>250</td>
  <td>0.48</td>
  <td>8.01</td>
</tr>

<tr>
  <td>0.1</td>
  <td>500</td>
  <td>0.99</td>
  <td>6.55</td>
</tr>

<tr>
  <td>0.5</td>
  <td>500</td>
  <td>0.99</td>
  <td>7.02</td>
</tr>

</table>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<div style="background-color: #C7FCE5; border-color:#9DFCD2  ; border-left: 5px solid #9DFCD2  ; padding: 0.5em;">
    <h4>Precisión</h4>
    <p>Centrándonos en la precisión, <b>ambos modelos de redes neuronales obtienen resultados parecidos en la mayoría de pruebas realizadas pero por el contrario se observa que numpy consigue unos tiempos de entrenamiento más rápidos que Keras</b>. Esto se debe a que <b>Keras implementa funciones más complejas en comparación a las funciones que hemos creado con Numpy</b> y, aunque esto proporciona a Keras que escale mejor para ejemplos más complejos, <b>provoca un mayor tiempo de aprendizaje del modelo</b>.</p>   
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<div style="background-color: #C7FCE5; border-color:#9DFCD2  ; border-left: 5px solid #9DFCD2  ; padding: 0.5em;">
    <h4>Velocidad de aprendizaje</h4>
    <p>La <b>velocidad de aprendizaje es un parámetro que controla cuánto estamos ajustando los pesos de nuestra red neuronal respecto al gradiente</b>. Cuanto más bajo sea el valor más lento correrán los valores dentro de la red. Aunque pueda parecer que una velocidad de aprendizaje baja puede conseguir mejores resultados, puede producir el efecto contrario ya que tiene el riesgo de caer en un mínimo local y tomar mucho tiempo para converger.</p>
    <p>Asímismo, ajustar la velocidad de aprendizaje en el proceso de entrenamiento es un factor muy importante para obtener una buena precisión. De esta manera, hemos aplicado los valores <b>10, 0,5 y 0,1</b> de tasa de aprendizaje sobre <b>numpy</b> y <b>keras</b>. Es fácil comprobar que para altos valores de este parámetro, como el valor 10, provoca una inestabilidad en la red y de hecho obtiene una precisión del 0,48, por el contrario los valores 0,5 y 0,1 funcionan mejor que el parámetro anterior. Sin embargo, tampoco se observa que para un valor muy bajo de velocidad de aprendizaje como lo es el 0,1, en el caso de numpy, no termina de converger. </p>
    <p>De esta manera, <b>parece ser que es interesante utilizar el valor intermedio de 0,5 ya que consigue converger mejor que con los otros parámetros.</b></p>
    <img src="learning-rate-graphic.png" alt="Diagrama del entrenamiento de la red neuronal"/>
<i>Gráfica de la comparación de diferentes parámetros de velocidad de aprendizaje para una red neuronal con las siguientes capas: [100, 80, 50, 40, 20, 5, 1]</i>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<div style="background-color: #C7FCE5; border-color:#9DFCD2  ; border-left: 5px solid #9DFCD2  ; padding: 0.5em;">
    <h4>Iteraciones (epoch)</h4>
    <p>En redes neuronales es necesario pasar el conjunto de datos varias veces a la misma red neuronal para su entrenamiento. El problema es que a priori es difícil determinar el valor óptimo para conseguir un mejor aprendizaje. Lo ideal es detener el número de iteraciones cuando el algoritmo se estanca y no consigue mejorar a continuación.</p>   
    <p>En nuestro caso hemos hecho pruebas con los valores de 50 iter., 250 iter. y 500 iteraciones. Es fácil darse cuenta que a partir de las 250 iteraciones el modelo ya no consigue mejores resultados y puede provocar un posible efecto de sobreaprendizaje. Por otro lado, 50 iteraciones son pocas y provoca resultado de precisión más bajos tanto en Keras como en Numpy.</p>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<div style="background-color: #C7FCE5; border-color:#9DFCD2  ; border-left: 5px solid #9DFCD2  ; padding: 0.5em;">
    <h4>Capas</h4>
    <p>Añadiendo capas y neuronas a la red conseguimos un aumento de predicción y capacidad de separación de los datos en su clasificación pero por el contrario podemos provocar una sobreespecialización y un aumento del coste computacional en tiempo de entrenamiento.</p>
    <p>Se ha probado con una arquitectura de 3 capas <b>layers_dims = [100, 20, 5, 1]</b> y con <b>layers_dims = [100, 80, 50, 40, 20, 5, 1] una de 6 capas.</b> y por lo general podemos decir que se ha aumentado el tiempo de aprendizaje tanto en Keras como en Numpy. Respecto a la precisión, no podemos afirmar conseguir una mejora en comparación con la primera arquitectura por lo que quizás no nos interesa esta ya que puede generar sobreaprendizaje.</p>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="References">References<a class="anchor-link" href="#References">&#182;</a></h3><p>Parte del codigo utilizado para desarrollar esta práctica proviene del curso de Coursera <a href="https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning">"Neural networks and deep learning"</a></p>

</div>
</div>
</div>
    </div>
  </div>
</body>

 


</html>
