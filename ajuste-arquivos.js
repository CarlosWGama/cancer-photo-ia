fs = require('fs')

const diretorio = './training/leucoplasia/positivo';
let total = 1;

fs.readdir(diretorio, (error, arquivos) => {
    
    arquivos.forEach((arquivo) => {
        const ext = arquivo.split('.').pop()
        const antigoNome = `${diretorio}/${arquivo}`;
        const novoNome = `${diretorio}/${total}.${ext}`;

        console.log(antigoNome)
        console.log(novoNome)

        fs.rename(antigoNome, novoNome, (erro) => console.log(erro))
        total++;
    })
})
