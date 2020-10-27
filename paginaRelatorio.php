<?php
        session_start();

        require_once 'banco.php';
        $banco = new Banco();
        /*
         * chama duas vezes(solução provisoria para atualizar o vetor de disciplina)
         * inseri um if para checar se é a primeira vez que executa mas mesmo assim não funciona
         * se retirar o if caso algum atualize a pagina vai marcar como feito e se não chamar
         * duas vezes o usuario tem que da f5 para mostrar as perguntas corretas
         */
        $banco->exibirPergunta();
        $saida = $banco->exibirPergunta();
        ?>
<html>
    <head>
        <meta charset="utf-8">
        <title>Pesquisa CPA</title>

        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" media="(max-width: 640px)" href="max-640px.css">
        <link rel="stylesheet" media="(min-width: 640px)" href="min-640px.css"> 
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css" rel="stylesheet">
        <link rel="stylesheet" href="lib/bootstrap/css/bootstrap.min.css" />
        <link rel="stylesheet" href="css/estilo3.css" />
        <script src="lib/jquery/jquery.min.js"></script>
        <script src="lib/bootstrap/js/bootstrap.min.js"></script>
    </head>
    <body background="img/fundoPagina.png">
        <header>
            <div>
                <img class="logo" src="img/CPA_Logo_UAM.png" />
            </div>

        </header>
        <form action="paginaRelatorio_resp.php" method="post">
            <input type="text" name="bloco">
            <input type="submit" value="Relatorio">
        </form>
    </body>
</html>
