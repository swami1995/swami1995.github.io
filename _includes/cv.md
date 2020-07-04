 I am a first year PhD student at the Robotics Institute at CMU and am broadly interested in machine learning problems faced by robotics systems. I like to seek solutions to these problems by looking at them from a more foundational lens of optimization and graphical models, sometimes taking inspiration from neuroscience and cognitive psychology. I previously finished my Masters in Robotics from <b>Robotics Institute</b> and was advised by [Prof. Katia Sycara](https://www.ri.cmu.edu/ri-faculty/katia-sycara/). My publications and descriptions of some selected projects are available below and on [my Google Scholar page](https://scholar.google.com/citations?user=do8COWIAAAAJ&hl=en). I also recently interned at [Nuro](https://nuro.ai/), self-driving car startup, where I worked on developing a differentiable planning pipeline.


## <i class="fa fa-chevron-right"></i> Education

<table class="table table-hover">
  <tr>
    <td class="col-md-3">June 2020 - Present</td>
    <td>
        <strong>Ph.D. in Robotics Research</strong>
          (0.00/0.00)
        <br>
      Carnegie Mellon University
    </td>
  </tr>
  <tr>
    <td class="col-md-3">Aug 2017 - Dec 2019</td>
    <td>
        <strong>M.S. in Robotics Research</strong>
          (4.10/4.33)
        <br>
      Carnegie Mellon University
    </td>
  </tr>
  <tr>
    <td class="col-md-3">July 2013 - May 2017</td>
    <td>
        <strong>B.S. in Electrical Engineering</strong>
          (8.99/10.00)
        <br>
      IIT-BHU Varanasi
    </td>
  </tr>
</table>


<!-- ## <i class="fa fa-chevron-right"></i> Experience
<table class="table table-hover">
<tr>
  <td class='col-md-3'>May 2019 - Present</td>
  <td><strong>Facebook AI</strong>, Research Scientist</td>
</tr>
<tr>
</tr>
<tr>
  <td class='col-md-3'>June 2018 - Sept 2018</td>
  <td><strong>Intel Labs</strong>, Research Intern</td>
</tr>
<tr>
</tr>
<tr>
  <td class='col-md-3'>May 2017 - Oct 2017</td>
  <td><strong>Google DeepMind</strong>, Research Intern</td>
</tr>
<tr>
</tr>
<tr>
  <td class='col-md-3'>May 2014 - Aug 2014</td>
  <td><strong>Adobe Research</strong>, Data Scientist Intern</td>
</tr>
<tr>
</tr>
<tr>
  <td class='col-md-3'>Dec 2013 - Jan 2014</td>
  <td><strong>Snowplow Analytics</strong>, Software Engineer Intern</td>
</tr>
<tr>
</tr>
<tr>
  <td class='col-md-3'>May 2013 - Aug 2013</td>
  <td><strong>Qualcomm</strong>, Software Engineer Intern</td>
</tr>
<tr>
</tr>
<tr>
  <td class='col-md-3'>May 2012 - Aug 2012</td>
  <td><strong>Phoenix Integration</strong>, Software Engineer Intern</td>
</tr>
<tr>
</tr>
<tr>
  <td class='col-md-3'>Jan 2011 - Aug 2011</td>
  <td><strong>Sunapsys</strong>, Network Administrator Intern</td>
</tr>
<tr>
</tr>
</table> -->


## <i class="fa fa-chevron-right"></i> Publications and Selected Projects <i class="fa fa-code-fork" aria-hidden="true"></i>

<a href="https://scholar.google.com/citations?user=do8COWIAAAAJ&hl=en&oi=sra" class="btn btn-primary" style="padding: 0.3em;">
  <i class="ai ai-google-scholar"></i> Google Scholar
</a>

<table class="table table-hover">
<tr>
<td class="col-md-3"><a href='https://arxiv.org/abs/1911.04024' target='_blank'><img src="images/publications/mame2019.png"/></a> </td>
<td>
    <strong>MAME : Model Agnostic Meta Exploration</strong><br>
    <strong>Swaminathan Gurumurthy</strong>, Sumit Kumar, Katia Sycara<br>
    CoRL 2019<br>
    
    [1] 
[<a href='https://arxiv.org/abs/1911.04024' 
    target='_blank'>pdf</a>] [<a href='https://github.com/swami1995/exp_maml/tree/another_sparse_branch_ppo' target='_blank'>code</a>] <br>
    
<!-- <div id="abs_amos2020differentiable" style="text-align: justify; display: none" markdown="1"> -->
We propose to explicitly model a separate exploration policy for the task distribution in Meta-RL given the requirements on sample efficiency. Having two different policies gives more flexibility during training and makes adaptation to any specific task easier. We show that using self-supervised or supervised learning objectives for adaptation stabilizes the training process and improves performance.
<!-- </div> -->

</td>
</tr>


<tr>
<td class="col-md-3"><a href='https://arxiv.org/abs/1808.04359' target='_blank'><img src="images/publications/visdial2019.png"/></a> </td>
<td>
    <strong>Community Regularization of Visually-Grounded Dialog</strong><br>
    Akshat Agarwal*, <strong>Swaminathan Gurumurthy*</strong>, Vasu Sharma*, Katia Sycara, Michael Lewis<br>
    AAMAS 2019 <strong>[Oral talk]</strong><br>
    
    [2] 
[<a href='https://arxiv.org/abs/1808.04359' 
    target='_blank'>pdf</a>] [<a href='https://github.com/agakshat/visualdialog-pytorch' target='_blank'>code</a>] <br>
    
<!-- <div id="abs_amos2020differentiable" style="text-align: justify; display: none" markdown="1"> -->
We aim to train 2 agents on the visual dialogue dataset where one agent is given access to an image and the other agent is tasked with guessing the contents of the image by establishing a dialogue with the first agent. The two agents are initially trained using supervision followed by Reinforce. In order to combat the resulting drift from natural language when training with Reinforce, we introduce a community regularization scheme of training a population of agents.
<!-- </div> -->

</td>
</tr>


<tr>
<td class="col-md-3"><a href='https://arxiv.org/abs/1807.03407' target='_blank'><img src="images/publications/pcc2019.png"/></a> </td>
<td>
    <strong> 3D Point Cloud Completion using Latent Optimization in GANs </strong><br>
    Shubham Agarwal*, <strong>Swaminathan Gurumurthy*</strong> <br>
    WACV 2019<br>
    
    [3] 
[<a href='https://arxiv.org/abs/1807.03407' target='_blank'>pdf</a>] <br>
    
<!-- <div id="abs_amos2020differentiable" style="text-align: justify; display: none" markdown="1"> -->
We address a fundamental problem with Neural Network based point cloud completion methods which reconstruct the entire structure rather than preserving the points already provided as input. These methods struggle when tested on unseen deformities. We address this problem by introducing a GAN based Latent optimization procedure to perform output constrained optimization using the regions provided in the input.
<!-- </div> -->

</td>
</tr>



<tr>
<td class="col-md-3"><a href='https://arxiv.org/abs/1805.05356' target='_blank'><img src="images/publications/compass2018.png"/></a> </td>
<td>
    <strong>Exploiting Data and Human Knowledge for Predicting Wildlife Poaching</strong><br>
    <strong>Swaminathan Gurumurthy</strong>, Lantao Yu, Chenyan Zhang, Yongchao Jin, Weiping Li, Haidong Zhang, Fei Fang<br>
    COMPASS 2019 <strong>[Oral talk]</strong><br>

    [4] 
[<a href='https://arxiv.org/abs/1805.05356' 
    target='_blank'>pdf</a>] [<a href='https://github.com/swami1995/PAWS-COMPASS' target='_blank'>code</a>] <br>
    
<!-- <div id="abs_amos2020differentiable" style="text-align: justify; display: none" markdown="1"> -->
Using past data of traps/snares found in a wildlife Sanctuary, we predict the regions of high probability of traps/snares to guide the rangers to patrol those regions. We use novel frameworks of incorporating expert domain knowledge for the dynamic sampling of data points in order to tackle the imbalance in data. We further use these regions to produce optimal patrol routes for the rangers. This has now been deployed in a conservation area in China.
<!-- </div> -->

</td>
</tr>


<tr>
<td class="col-md-3"><a href='http://openaccess.thecvf.com/content_cvpr_2017/papers/Gurumurthy_DeLiGAN__Generative_CVPR_2017_paper.pdf' target='_blank'><img src="images/publications/deligan2017.png"/></a> </td>
<td>
    <strong> DeLiGAN: GANs for Diverse and Limited Data </strong><br>
    <strong>Swaminathan Gurumurthy*</strong>, Ravi Kiran S.* and R. Venkatesh Babu <br>
    CVPR 2017<br>

    [5] 
[<a href='http://openaccess.thecvf.com/content_cvpr_2017/papers/Gurumurthy_DeLiGAN__Generative_CVPR_2017_paper.pdf' 
    target='_blank'>pdf</a>] [<a href='https://github.com/val-iisc/deligan' target='_blank'>code</a>] <br>
    
<!-- <div id="abs_amos2020differentiable" style="text-align: justify; display: none" markdown="1"> -->
We try to explore the idea of finding high probability regions in the latent space of GANs by learning a latent space representation using learnable Mixture of Gaussians. This enables the GAN to model a multimodal distribution and stabilizes training as observed visually and by the intra-class variance measured using a modified inception score. Our modification is especially useful when the dataset is very small and diverse.
<!-- </div> -->

</td>
</tr>

<tr>
<td class="col-md-3"><a href='https://drive.google.com/file/d/1l8ngpk-A3E01gTBhAsXT7IzV_bIJ_b12/view?usp=sharing' target='_blank'><img src="images/publications/bbox_attack.png"/></a> </td>
<td>
    <strong> Query Efficient Black Box Attacks in Neural Networks </strong><br>
    <strong>Swaminathan Gurumurthy</strong>, Fei Fang and Martial Hebert <br>

    [6] 
[<a href='https://drive.google.com/file/d/1l8ngpk-A3E01gTBhAsXT7IzV_bIJ_b12/view?usp=sharing' 
    target='_blank'>pdf</a>] <br>
    
<!-- <div id="abs_amos2020differentiable" style="text-align: justify; display: none" markdown="1"> -->
We test various methods to increase the sample efficiency of adversarial black box attacks on Neural nets. In one of the methods, we analyze the transferability of gradients and find that it has two components: Network specific components and Task specific components. The task specific component corresponds to the transferable properties of adversarial examples between architectures. Hence, we attempted to isolate this component and enhance the transfer properties. We then perform multiple queries on the black box network to obtain the architecture specific components using ES.
<!-- </div> -->

</td>
</tr>


<tr>
<td class="col-md-3"><img src="images/publications/slam.png"/></td>
<td>
    <strong> Visual SLAM based SfM for Boreholes </strong><br>
    <strong>Swaminathan Gurumurthy</strong>, Tat-Jun Chin and Ian Reid <br>

    [7] 
[<a href='https://github.com/swami1995/vo-slam' target='_blank'>code</a>] <br>
    
<!-- <div id="abs_amos2020differentiable" style="text-align: justify; display: none" markdown="1"> -->
Built a package to construct a sparse map and camera trajectory using SIFT features, fine-tuned using bundle adjustment and loop closure. It was tailored for boreholes and underground scenes with forward motion, where most of the current state of the art approaches like LSD SLAM, ORB SLAM and SVO struggled at both localization and mapping.
<!-- </div> -->

</td>
</tr>

<tr>
<td class="col-md-3"><img src="images/publications/explore2018.png"/> </td>
<td>
    <strong> Off-on policy learning </strong><br>
    <strong>Swaminathan Gurumurthy</strong>, Bhairav Mehta, Anirudh Goyal and Yoshua Bengio <br>

    [8] 
    
<!-- <div id="abs_amos2020differentiable" style="text-align: justify; display: none" markdown="1"> -->
On policy methods are known to exhibit stable behavior and off-policy methods are known to be sample efficient. The goal here was to get the best of both worlds. We first developed a self-imitation based method to learn from a diverse set of exploratory policies which perform coordinated exploration. We also tried a meta-learning objective to ensure that the off-policy updates to the policies are aligned with future on-policy updates. This leads to more stable training but fails to reach peak performance in most continuous control tasks we tested on.
<!-- </div> -->

</td>
</tr>

</td>
</tr>

<tr>
<td class="col-md-3"><img src="images/publications/interpretability2018.png"/>> </td>
<td>
    <strong> Exploring interpretability in Atari Games for RL policies using Counterfactuals </strong><br>
    <strong>Swaminathan Gurumurthy</strong>, Akshat Agarwal, Prof. Katia Sycara <br>

    [9] 
    
<!-- <div id="abs_amos2020differentiable" style="text-align: justify; display: none" markdown="1"> -->
We aimed to understand what RL agents learn in simple games such as in Atari. We developed a GAN based method to find counterfactuals for the policies, i.e., we find small perturbations in the scene that can lead to changes in the agent action and use these to interpret agent behavior. GAN in this case is used to avoid adversarial examples and produce semantically meaningful perturbations.
<!-- </div> -->

</td>
</tr>


</table>
