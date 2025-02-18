- How to copy the repository on your machine using GIT?

  - Step 1: Copy the ssh public key of the machine on which you want to clone the repository
  To do so, type on the computer you want to use :
  cd ~/
  cat .ssh/id_rsa.pub
  This generates a ssh public key (= a few lines of alpha numerical characters)

  If you get the error message « cat: .ssh/id_rsa.pub: No such file or directory », it means you do not already have a public ssh key and you need to generate it with the command:
  ssh-keygen -t rsa

  Then restart the procedure

  - Step 2: Copy the public key on Github
  To do so, connect to Github. 
  Then go on your profile (top-right corner).
  Click on "Settings"
  Click on "SSH and GPG keys"
  Click on "New SSH key"
  Enter a title and copy your public key


  - Step 3: Do a copy via GIT
  Type "git clone git@github.com:pdonnel/EC_exercise.git " on the machine